import argparse, csv, math
import numpy as np
from scipy.spatial import cKDTree
from sgp4.api import jday, SatrecArray
import WalkerDeltaSGP

parser = argparse.ArgumentParser()
parser.add_argument('--cam_file', type=str, required=True)
parser.add_argument('--baseline_file', type=str, required=True)
parser.add_argument('--num_sats', type=int, required=True)
parser.add_argument('--num_planes', type=int, required=True)
parser.add_argument('--output', type=str, default='cam_effects.csv')
args = parser.parse_args()

mu = 398600.4418
dt_coarse = 30
screening_radius = 30
jd_start, fr_start = jday(2026, 5, 28, 0, 0, 0.0)

# Generate constellation
argConst = {
    'name': f'Constellation-{args.num_sats}',
    'sma': 6921000., 'inc': 53., 'firstRAAN': 0.,
    't': args.num_sats, 'p': args.num_planes, 'f': 1
}
wc = WalkerDeltaSGP.WalkerDeltaSGP(argConst)
satellites = wc.generate()
sat_array = SatrecArray(satellites)

sma_km = argConst['sma'] / 1000.0
n = math.sqrt(mu / sma_km**3)
T = 2 * math.pi / n
n_orbits_before = 2

# Load baseline distances
baseline = {}
with open(args.baseline_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (int(row['sat_i']), int(row['sat_j']))
        baseline[key] = float(row['min_dist_km'])
print(f"Loaded {len(baseline)} baseline pairs")

column_headers = [
    'obj_i',
    'obj_j',
    'miss_km',
    'Pc',
    'dv_ms',
    'type',
    'tca',
    'cam_sat_idx'
]

maneuvers = []
with open(args.cam_file) as f:
    # 1. Pass the headers list directly
    reader = csv.DictReader(f, fieldnames=column_headers)
    
    for row in reader:
        # 2. If the stray header line appears anywhere in the file, skip it
        if row['dv_ms'] == 'dv_ms':
            continue
            
        dv = float(row['dv_ms'])
        if dv <= 0:
            continue
            
        tca = float(row['tca'])
        maneuvers.append({
            'sat_idx': int(row['cam_sat_idx']),
            't_burn': tca - n_orbits_before * T,
            't_return': tca,
            'dv_km': dv / 1000.0 / 2,
        })
print(f"Loaded {len(maneuvers)} maneuvers")

# Determine simulation window: cover all CAM windows with padding
if not maneuvers:
    print("No maneuvers to analyze.")
    exit()

sim_start = max(0, min(m['t_burn'] for m in maneuvers) - 60)
sim_end = max(m['t_return'] for m in maneuvers) + 60
num_steps = int((sim_end - sim_start) / dt_coarse) + 1

print(f"Simulation window: {sim_start:.0f}s to {sim_end:.0f}s ({(sim_end-sim_start)/3600:.1f} hrs)")

# For each CAM, collect all degraded crossings during its active window
cam_effects = []
chunk_size = 500

# Build time arrays for just the simulation window
times_fr_local = fr_start + (sim_start + np.arange(num_steps) * dt_coarse) / 86400.0
times_jd_local = np.full(num_steps, jd_start)

# Per-timestep events: (time, cam_sat, neighbor_sat, perturbed_dist, baseline_dist)
all_events = []

print(f"Screening {num_steps} timesteps...")
for chunk_start in range(0, num_steps, chunk_size):
    chunk_end = min(chunk_start + chunk_size, num_steps)
    if chunk_start % 2000 == 0:
        print(f"  Step {chunk_start}/{num_steps}, events: {len(all_events)}")

    err, pos, vel = sat_array.sgp4(
        times_jd_local[chunk_start:chunk_end],
        times_fr_local[chunk_start:chunk_end]
    )

    for s in range(pos.shape[1]):
        step = chunk_start + s
        t = sim_start + step * dt_coarse

        p_nominal = pos[:, s, :].copy()
        v = vel[:, s, :]
        e = err[:, s]
        valid = (e == 0) & np.all(np.isfinite(p_nominal), axis=1)

        # Find which CAMs are active at this timestep
        active_cams = []
        p_perturbed = p_nominal.copy()

        for mnvr in maneuvers:
            idx = mnvr['sat_idx']
            if idx >= len(satellites) or not valid[idx]:
                continue
            if mnvr['t_burn'] <= t <= mnvr['t_return']:
                dt_since = t - mnvr['t_burn']
                displacement = 3 * mnvr['dv_km'] * dt_since
                v_hat = v[idx] / np.linalg.norm(v[idx])
                p_perturbed[idx] += displacement * v_hat
                active_cams.append(mnvr)

        if not active_cams:
            continue

        # Screen perturbed positions of CAM'd sats against nominal positions
        cam_sat_ids = set(m['sat_idx'] for m in active_cams)

        # Build tree of nominal positions (non-CAM'd satellites only)
        nominal_mask = valid.copy()
        for idx in cam_sat_ids:
            nominal_mask[idx] = False
        nominal_pos = p_nominal[nominal_mask]
        nominal_idx = np.where(nominal_mask)[0]

        if len(nominal_pos) == 0:
            continue

        tree = cKDTree(nominal_pos)

        for mnvr in active_cams:
            cam_idx = mnvr['sat_idx']
            perturbed_point = p_perturbed[cam_idx]
            neighbors = tree.query_ball_point(perturbed_point, screening_radius)

            for n_local in neighbors:
                neighbor_idx = int(nominal_idx[n_local])
                perturbed_dist = np.linalg.norm(perturbed_point - nominal_pos[n_local])
                pair_key = (min(cam_idx, neighbor_idx), max(cam_idx, neighbor_idx))
                baseline_dist = baseline.get(pair_key, 999.0)

                # Only record if this is worse than baseline
                if perturbed_dist < baseline_dist:
                    all_events.append({
                        'time': t,
                        'cam_sat': cam_idx,
                        'cam_tca': mnvr['t_return'],
                        'cam_dv': mnvr['dv_km']*1000,
                        'neighbor_sat': neighbor_idx,
                        'perturbed_dist': perturbed_dist,
                        'baseline_dist': baseline_dist,
                        'degradation': baseline_dist - perturbed_dist,
                    })

    del err, pos, vel

print(f"\nTotal degraded crossing events: {len(all_events)}")

# Group events by CAM (identified by cam_sat + cam_tca)
from collections import defaultdict
cam_groups = defaultdict(list)
for event in all_events:
    cam_key = (event['cam_sat'], event['cam_tca'])
    cam_groups[cam_key].append(event)

# Write per-CAM summary
with open(args.output, 'w') as f:
    f.write('cam_sat,cam_tca,cam_dv_ms,num_degraded_crossings,'
            'num_unique_neighbors,worst_dist_km,worst_degradation_km,'
            'mean_degradation_km\n')
    for (cam_sat, cam_tca), events in sorted(cam_groups.items()):
        unique_neighbors = set(e['neighbor_sat'] for e in events)
        worst_dist = min(e['perturbed_dist'] for e in events)
        worst_deg = max(e['degradation'] for e in events)
        mean_deg = np.mean([e['degradation'] for e in events])
        f.write(f"{cam_sat},{cam_tca:.2f},{events[0]['cam_dv']:.4f},"
                f"{len(events)},{len(unique_neighbors)},"
                f"{worst_dist:.6f},{worst_deg:.6f},{mean_deg:.6f}\n")

print(f"Wrote {len(cam_groups)} CAM effect summaries to {args.output}")

# Also write raw events for detailed analysis
raw_output = args.output.replace('.csv', '_raw.csv')
with open(raw_output, 'w') as f:
    f.write('time_s,cam_sat,cam_tca,neighbor_sat,perturbed_dist_km,baseline_dist_km,degradation_km\n')
    for e in sorted(all_events, key=lambda x: x['perturbed_dist']):
        f.write(f"{e['time']:.0f},{e['cam_sat']},{e['cam_tca']:.2f},"
                f"{e['neighbor_sat']},{e['perturbed_dist']:.6f},"
                f"{e['baseline_dist']:.6f},{e['degradation']:.6f}\n")

print(f"Wrote {len(all_events)} raw events to {raw_output}")