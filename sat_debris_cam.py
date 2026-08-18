import WalkerDeltaSGP
from sgp4.api import jday
import numpy as np
import math
from scipy.spatial import KDTree
from scipy.optimize import minimize_scalar, brentq
from spacetrack import SpaceTrackClient
import spacetrack.operators as op
from sgp4.api import Satrec, SatrecArray
from scipy.integrate import dblquad
import sys, os
import argparse

# Config
chunk_size = 500
screening_radius = 200
refine_threshold = 50
hard_body_radius = 0.01
dt_coarse = 5
dt_fine = 0.01
sim_time = 30 * 24 * 60 * 60
jd_start, fr_start = jday(2026, 5, 28, 0, 0, 0.0)
dt_days = dt_coarse / 86400.0
mu = 398600.435507

from nasa_sbm import generate_fragments

# Iridium-Cosmos parameters
m_iridium = 560    # kg
m_cosmos = 900     # kg
v_impact = 11.7    # km/s

R_earth = 6371.0

# You need a collision position and velocity for each altitude
# Use a circular orbit velocity at each altitude
scenarios = {
    'at_altitude':   550,  # km
    #'above_50km':    600,  # km
    #'below_50km':    500,  # km
}

for label, alt in scenarios.items():
    r = R_earth + alt
    v_circ = math.sqrt(mu / r)
    
    # Collision at a point on the orbit (arbitrary position)
    collision_pos = np.array([r, 0, 0])          # km, TEME
    collision_vel = np.array([0, v_circ, 0])     # km/s, TEME
    
    fragments = generate_fragments(
        m_target=m_cosmos,
        m_projectile=m_iridium,
        v_impact=v_impact,
        parent_pos=collision_pos,
        parent_vel=collision_vel,
        jd_epoch=jd_start,
        fr_epoch=fr_start,
        min_size=0.05,
        max_fragments=5000,
        seed=42
    )


sigma_radial_sat  = 0.05  
sigma_along_sat   = 0.3    # (SOCRATES value)
sigma_cross_sat   = 0.1   

sigma_radial_deb  = 0.2    
sigma_along_deb   = 1.0    
sigma_cross_deb   = 0.2    

parser = argparse.ArgumentParser()
parser.add_argument('--num_sats', type=int, required=True)
parser.add_argument('--num_planes', type=int, required=True)
parser.add_argument('--output', type=str, default='results.csv')
parser.add_argument('--start_day', type=int, default=0)
parser.add_argument('--end_day', type=int, default=30)
parser.add_argument('--debris_file', type=str, default='debris_catalog.tle')
args = parser.parse_args()

argConst = {
    'name': f'Constellation-{args.num_sats}',
    'sma': 6921000.,
    'inc': 53.,
    'firstRAAN': 0.,
    't': args.num_sats,
    'p': args.num_planes,
    'f': 1
}

def size_mb(obj):
    return sys.getsizeof(obj) / 1e6

if __name__ == '__main__':
    # Objects
    wc = WalkerDeltaSGP.WalkerDeltaSGP(argConst)
    satellites = wc.generate()

    with open(args.debris_file, 'r') as f:
        debris_tles = f.read()
    lines = debris_tles.strip().split('\n')
    debris_objects = []

    for i in range(0, len(lines), 2):
        line1 = lines[i]
        line2 = lines[i + 1]
        sat = Satrec.twoline2rv(line1, line2)
        error, pos, vel = sat.sgp4(jd_start, fr_start) 
        if sat.error == 0 and all(math.isfinite(p) for p in pos) and (pos[0]**2 + pos[1]**2 + pos[2]**2) > 1000:
            debris_objects.append(sat)

    print(f"Loaded {len(debris_objects)} debris objects")

    objects = satellites + debris_objects + fragments
    objects_array = SatrecArray(objects)
    constellation_ids = set(range(len(satellites)))
    debris_ids = set(range(len(satellites), len(objects)))

    T_orbit = 2 * math.pi * math.sqrt((argConst['sma']/1000)**3 / mu)  # seconds
    min_separation = T_orbit / 2
    pair_latest_time = {}

    jd = jd_start

    num_steps = sim_time // dt_coarse
    times_fr = fr_start + np.arange(num_steps) * dt_coarse / 86400.0
    times_jd = np.full(num_steps, jd_start)
    debris_mask = np.array([i in debris_ids for i in range(len(objects))])
    conjunctions = {}

    del debris_tles, lines
    del satellites, debris_objects

    window_days = 3
    window_steps = (window_days * 86400) // dt_coarse

    start_step = int(args.start_day * 86400 / dt_coarse)
    end_step = int(args.end_day * 86400 / dt_coarse)

    for window_start in range(start_step, end_step, window_steps):
        day = int(window_start * dt_coarse // 86400)
        
        print(f"Window starting on day {window_start*dt_coarse//86400}")
        conjunctions = {}       # reset each window
        pair_latest_time = {}
        refined_conjunctions = []
        window_end = min(window_start + window_steps, num_steps)
        print(f"Screening {window_steps} timesteps in chunks of {chunk_size}")
        for chunk_start in range(window_start, window_end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, window_end)
            print(f"  Chunk {chunk_start}-{chunk_end} of {window_steps}")            
            err, pos, vel = objects_array.sgp4(
                times_jd[chunk_start:chunk_end],
                times_fr[chunk_start:chunk_end]
            )
            
            for s in range(pos.shape[1]):
                step = chunk_start + s
                t = step * dt_coarse
                p = pos[:, s, :]
                e = err[:, s]
                
                valid = (e == 0) & np.all(np.isfinite(p), axis=1) & (np.sum(p**2, axis=1) > 1e6)
                valid_pos = p[valid]
                valid_idx = np.where(valid)[0]
                
                tree = KDTree(valid_pos)
                close_pairs = tree.query_pairs(screening_radius)
                
                for (i, j) in close_pairs:
                    ri, rj = int(valid_idx[i]), int(valid_idx[j])
                    if debris_mask[ri] == debris_mask[rj]:
                        continue
                    dist = np.linalg.norm(valid_pos[i] - valid_pos[j])
                    pair = (min(ri, rj), max(ri, rj))
                    
                    if pair in pair_latest_time:
                        last_t, last_event = pair_latest_time[pair]
                        if t - last_t > min_separation:
                            event_idx = last_event + 1
                            key = (pair[0], pair[1], event_idx)
                            conjunctions[key] = {
                                'min_dist': dist, 'time': t,
                                'obj_i': pair[0], 'obj_j': pair[1]
                            }
                            pair_latest_time[pair] = (t, event_idx)
                        else:
                            key = (pair[0], pair[1], last_event)
                            if key in conjunctions and dist < conjunctions[key]['min_dist']:
                                conjunctions[key]['min_dist'] = dist
                                conjunctions[key]['time'] = t
                            pair_latest_time[pair] = (t, last_event)
                    else:
                        event_idx = 0
                        key = (pair[0], pair[1], event_idx)
                        conjunctions[key] = {
                            'min_dist': dist, 'time': t,
                            'obj_i': pair[0], 'obj_j': pair[1]
                        }
                        pair_latest_time[pair] = (t, event_idx)
            del err, pos, vel
            

        print(f"Found {len(conjunctions)} unique pairs")

        sat_debris_conjs = []
        for key, data in conjunctions.items():
            ri, rj = data['obj_i'], data['obj_j']
            if (ri in constellation_ids) or (rj in constellation_ids):  # one or the other
                sat_debris_conjs.append(data)

        sat_debris_conjs.sort(key=lambda x: x['min_dist'])
        print(f"Sat-debris pairs: {len(sat_debris_conjs)}")
            
        # Refine close approaches
        refined_conjunctions = []

        candidates = {k: v for k, v in conjunctions.items() if v['min_dist'] <= refine_threshold}
        print(f"Coarse candidates: {len(candidates)}")

        # Stage 1: quick sub filter
        quick_candidates = {}
        for idx, (key, coarse_data) in enumerate(candidates.items()):
            if idx % 10000 == 0:
                print(f"  Quick filter {idx}/{len(candidates)}")
            
            oi = coarse_data['obj_i']
            oj = coarse_data['obj_j']
            t_center = coarse_data['time']
            
            min_dist = coarse_data['min_dist']
            best_t = t_center
            
            # Check 24 steps in 120s window
            for dt in range(-60, 61, 5):
                t_check = t_center + dt
                fr = fr_start + t_check / 86400.0
                e1, p1, _ = objects[oi].sgp4(jd_start, fr)
                e2, p2, _ = objects[oj].sgp4(jd_start, fr)
                if e1 != 0 or e2 != 0:
                    continue
                d = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
                if d < min_dist:
                    min_dist = d
                    best_t = t_check
            
            if min_dist < 10.0:  # only keep close approaches
                coarse_data['min_dist'] = min_dist
                coarse_data['time'] = best_t
                quick_candidates[key] = coarse_data

        print(f"After quick filter: {len(quick_candidates)}")
        del conjunctions

        # Stage 2: full refinement
        refined_conjunctions = []
        for idx, (key, coarse_data) in enumerate(quick_candidates.items()):
            if idx % 100 == 0:
                print(f"  Refining {idx}/{len(quick_candidates)}")
            
            oi = coarse_data['obj_i']
            oj = coarse_data['obj_j']
            
            t_window = (-dt_coarse, dt_coarse)
            
            def distance_at_t(t, _i=oi, _j=oj, _ct=coarse_data['time']):
                fr = fr_start + (_ct + t) / 86400.0
                dum, pos_i, dum2 = objects[_i].sgp4(jd_start, fr)
                dum, pos_j, dum2 = objects[_j].sgp4(jd_start, fr)
                return np.linalg.norm(np.array(pos_i) - np.array(pos_j))
            
            result = minimize_scalar(distance_at_t, bounds=t_window, method='bounded')
            tca = result.x
            miss_distance = result.fun

            if miss_distance < refine_threshold:
                fr = fr_start + (coarse_data['time'] + tca) / 86400.0
                jd = jd_start

                dum, pos_i, vel_i = objects[oi].sgp4(jd, fr)
                dum, pos_j, vel_j = objects[oj].sgp4(jd, fr)
                pos_i = np.array(pos_i)
                pos_j = np.array(pos_j)
                vel_i = np.array(vel_i)
                vel_j = np.array(vel_j)
                rel_pos = pos_j - pos_i
                rel_vel = vel_j - vel_i

                refined_conjunctions.append({
                    'obj_i': oi,
                    'obj_j': oj,
                    'tca': coarse_data['time'] + tca,
                    'miss_distance': miss_distance,
                    'rel_position': rel_pos,
                    'rel_velocity': rel_vel,
                    'pos_i': pos_i,
                    'pos_j': pos_j,
                    'vel_i': vel_i,
                    'vel_j': vel_j
                })

        # Tooling for collision probabilities

        def rtn_to_teme_rotation(pos, vel):
            pos = np.array(pos)
            vel = np.array(vel)
            
            r_hat = pos / np.linalg.norm(pos)    
            n_hat = np.cross(pos, vel)           
            n_hat = n_hat / np.linalg.norm(n_hat) 
            t_hat = np.cross(n_hat, r_hat)       
            
            return np.column_stack([r_hat, t_hat, n_hat])

        def build_covariance_teme(pos, vel, sig_r, sig_t, sig_n):
            C_rtn = np.diag([sig_r**2, sig_t**2, sig_n**2])
            R = rtn_to_teme_rotation(pos, vel)
            return R @ C_rtn @ R.T

        def Pc2D_Foster(r1, v1, cov1, r2, v2, cov2, HBR, RelTol=1e-8):
            """
            2D Probability of Collision (Foster method).
            """
            r1, v1, r2, v2 = [np.asarray(x, dtype=float) for x in [r1, v1, r2, v2]]
            cov1 = np.asarray(cov1, dtype=float)
            cov2 = np.asarray(cov2, dtype=float)

            covcomb = cov1[:3,:3] + cov2[:3,:3]

            r = r1 - r2
            v = v1 - v2
            h = np.cross(r, v)

            y_hat = v / np.linalg.norm(v)    
            z_hat = h / np.linalg.norm(h)    
            x_hat = np.cross(y_hat, z_hat)   

            eci2xyz = np.array([x_hat, y_hat, z_hat])

            covcomb_xyz = eci2xyz @ covcomb @ eci2xyz.T
            
            P = np.array([[1, 0, 0],
                        [0, 0, 1]])
            Cp = P @ covcomb_xyz @ P.T  # 2x2 covariance in encounter plane
            
            # Check positive definiteness
            eigvals = np.linalg.eigvalsh(Cp)
            if np.min(eigvals) <= 0:
                # Eigenvalue clipping remediation
                Lclip = (1e-4 * HBR)**2
                eigvals_clipped = np.maximum(eigvals, Lclip)
                eigvecs = np.linalg.eigh(Cp)[1]
                Cp = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
            
            Cp_det = np.linalg.det(Cp)
            Cp_inv = np.linalg.inv(Cp)
            
            x0 = np.linalg.norm(r)
            z0 = 0.0
            
            # Integration normalization
            norm_const = 1.0 / (2 * np.pi * np.sqrt(Cp_det))

            C = Cp_inv
            def integrand(z, x):
                return np.exp(-0.5 * (C[0,0]*x**2 + (C[0,1]+C[1,0])*x*z + C[1,1]*z**2))
            
            upper = lambda x: np.sqrt(max(0, HBR**2 - (x - x0)**2))
            lower = lambda x: -np.sqrt(max(0, HBR**2 - (x - x0)**2))
            
            Pc, _ = dblquad(integrand, x0 - HBR, x0 + HBR,
                            lower, upper,
                            epsabs=1e-13, epsrel=RelTol)
            
            Pc *= norm_const
            
            return Pc, Cp
        
        def compute_cam_delta_v(conj, cov_i, cov_j, sma_km, HBR,
                    Pc_threshold=1e-6, n_orbits_before=2):
            """
            Uses CW along-track drift + Foster Pc evaluation.
            """
            if conj['Pc'] <= Pc_threshold:
                return 0.0
        
            mu = 398600.4418
            n = math.sqrt(mu / sma_km**3)
            T = 2 * math.pi / n
            dt = n_orbits_before * T
            
            pos_i = conj['pos_i']
            vel_i = conj['vel_i']
            pos_j = conj['pos_j']
            vel_j = conj['vel_j']
            v_hat = vel_i / np.linalg.norm(vel_i)
            
            def Pc_after_maneuver(dv_ms):
                dv_km = dv_ms / 1000.0
                displacement = 3 * dv_km * dt
                pos_i_shifted = pos_i + displacement * v_hat
                Pc_new, _ = Pc2D_Foster(pos_i_shifted, vel_i, cov_i,
                                        pos_j, vel_j, cov_j, HBR)
                return Pc_new
            
            def f(dv_ms):
                return Pc_after_maneuver(dv_ms) - Pc_threshold
            
            try:
                dv_solution = brentq(f, 0.001, 5000, xtol=0.01)
                return 2 * dv_solution  # maneuver
            except ValueError:
                if Pc_after_maneuver(0.01) <= Pc_threshold:
                    return 2 * 0.01
                return 0.0  # give up


        sma_km = argConst['sma'] / 1000.0
        total_dv = 0.0
        maneuver_count = 0

        for idx, conj in enumerate(refined_conjunctions):  
            if idx % 50 == 0:
                print(f"  Pc/CAM {idx}/{len(refined_conjunctions)}")
            oi = conj['obj_i']
            oj = conj['obj_j']
            sig_r_i = sigma_radial_deb if oi in debris_ids else sigma_radial_sat
            sig_t_i = sigma_along_deb  if oi in debris_ids else sigma_along_sat
            sig_n_i = sigma_cross_deb  if oi in debris_ids else sigma_cross_sat
            
            sig_r_j = sigma_radial_deb if oj in debris_ids else sigma_radial_sat
            sig_t_j = sigma_along_deb  if oj in debris_ids else sigma_along_sat
            sig_n_j = sigma_cross_deb  if oj in debris_ids else sigma_cross_sat
            
            cov_i = build_covariance_teme(conj['pos_i'], conj['vel_i'], sig_r_i, sig_t_i, sig_n_i)
            cov_j = build_covariance_teme(conj['pos_j'], conj['vel_j'], sig_r_j, sig_t_j, sig_n_j)
            
            Pc, Cp = Pc2D_Foster(conj['pos_i'], conj['vel_i'], cov_i,
                                conj['pos_j'], conj['vel_j'], cov_j,
                                hard_body_radius)
            conj['Pc'] = Pc
            
            dv = compute_cam_delta_v(conj, cov_i, cov_j, sma_km, hard_body_radius)
            conj['delta_v_ms'] = dv
            
            if dv > 0:
                total_dv += dv
                maneuver_count += 1

        # Classify
        for conj in refined_conjunctions:
            i, j = conj['obj_i'], conj['obj_j']
            if i in constellation_ids and j in constellation_ids:
                conj['type'] = 'sat-sat'
            elif i in debris_ids and j in debris_ids:
                conj['type'] = 'debris-debris'
            else:
                conj['type'] = 'sat-debris'

        # Results
        refined_conjunctions.sort(key=lambda c: c['Pc'], reverse=True)

        # Save everything to file
        with open(args.output, 'a') as f:
            if day == 0:
                f.write('obj_i,obj_j,miss_km,Pc,dv_ms,type,tca,cam_sat_idx\n')
            for conj in refined_conjunctions:
                if conj['tca'] < 0:
                    continue
                
                if conj['obj_i'] in constellation_ids:
                    cam_sat = conj['obj_i']
                else:
                    cam_sat = conj['obj_j']
                
                f.write(f"{conj['obj_i']},{conj['obj_j']},"
                        f"{conj['miss_distance']:.6f},{conj['Pc']:.6e},"
                        f"{conj['delta_v_ms']:.4f},sat-debris,"
                        f"{conj['tca']:.6f},{cam_sat}\n")

        # Print summary
        print(f"\nTotal conjunctions: {len(refined_conjunctions)}")
        print(f"  Sat-sat: {sum(1 for c in refined_conjunctions if c['type']=='sat-sat')}")
        print(f"  Sat-debris: {sum(1 for c in refined_conjunctions if c['type']=='sat-debris')}")
        print(f"\nManeuvers: {maneuver_count}")
        print(f"Total delta-v: {total_dv:.4f} m/s")
        print(f"Mean per maneuver: {total_dv/max(1,maneuver_count):.4f} m/s")

        # Print only conjunctions with Pc > 0
        active = [c for c in refined_conjunctions if c['Pc'] > 0]
        print(f"\nTop {min(50, len(active))} non-zero Pc conjunctions:")
        for conj in active[:50]:
            print(f"  {conj['obj_i']:>5} {conj['obj_j']:>5}  "
                f"miss={conj['miss_distance']:.3f} km  "
                f"Pc={conj['Pc']:.2e}  "
                f"dv={conj['delta_v_ms']:.2f} m/s  "
                f"{conj['type']}")

    print(f"\nFull footlong results saved to {args.output}")
