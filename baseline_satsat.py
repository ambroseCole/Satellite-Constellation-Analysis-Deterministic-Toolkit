import WalkerDeltaSGP
from sgp4.api import jday
import numpy as np
import math
from scipy.spatial import cKDTree
from sgp4.api import SatrecArray
import sys, os
import argparse

# Config
chunk_size = 500
screening_radius = 30  # sat-sat only, low relative velocity
dt_coarse = 30
sim_time = 30 * 24 * 60 * 60
jd_start, fr_start = jday(2026, 5, 28, 0, 0, 0.0)
mu = 398600.4418

parser = argparse.ArgumentParser()
parser.add_argument('--num_sats', type=int, required=True)
parser.add_argument('--num_planes', type=int, required=True)
parser.add_argument('--output', type=str, default='baseline_satsat.csv')
args = parser.parse_args()

argConst = {
    'name': f'Constellation-{args.num_sats}',
    'sma': 6921000., 'inc': 53., 'firstRAAN': 0.,
    't': args.num_sats, 'p': args.num_planes, 'f': 1
}

if __name__ == '__main__':
    wc = WalkerDeltaSGP.WalkerDeltaSGP(argConst)
    satellites = wc.generate()
    sat_array = SatrecArray(satellites)

    num_steps = sim_time // dt_coarse
    times_fr = fr_start + np.arange(num_steps) * dt_coarse / 86400.0
    times_jd = np.full(num_steps, jd_start)

    # Track closest approach per unique pair
    conjunctions = {}

    print(f"Screening {num_steps} timesteps for baseline sat-sat distances...")
    for chunk_start in range(0, num_steps, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_steps)
        if chunk_start % 5000 == 0:
            print(f"  Step {chunk_start}/{num_steps}")

        err, pos, vel = sat_array.sgp4(
            times_jd[chunk_start:chunk_end],
            times_fr[chunk_start:chunk_end]
        )

        for s in range(pos.shape[1]):
            step = chunk_start + s
            t = step * dt_coarse
            p = pos[:, s, :]
            e = err[:, s]

            valid = (e == 0) & np.all(np.isfinite(p), axis=1)
            valid_pos = p[valid]
            valid_idx = np.where(valid)[0]

            tree = cKDTree(valid_pos)
            close_pairs = tree.query_pairs(screening_radius)

            for (i, j) in close_pairs:
                ri, rj = int(valid_idx[i]), int(valid_idx[j])
                dist = np.linalg.norm(valid_pos[i] - valid_pos[j])
                key = (min(ri, rj), max(ri, rj))
                if key not in conjunctions or dist < conjunctions[key]:
                    conjunctions[key] = dist

        del err, pos, vel

    # Write results
    with open(args.output, 'w') as f:
        f.write('sat_i,sat_j,min_dist_km\n')
        for (i, j), dist in sorted(conjunctions.items(), key=lambda x: x[1]):
            f.write(f"{i},{j},{dist:.6f}\n")

    print(f"Wrote {len(conjunctions)} pairs to {args.output}")
    print(f"Minimum baseline separation: {min(conjunctions.values()):.4f} km")