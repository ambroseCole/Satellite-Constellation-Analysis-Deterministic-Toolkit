import argparse, csv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--files', type=str, nargs='+', required=True)
parser.add_argument('--num_sats', type=int, required=True)
args = parser.parse_args()

# Load all CAM effect files
cam_effects = []
for filepath in args.files:
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cam_effects.append({
                'cam_sat': int(row['cam_sat']),
                'cam_tca': float(row['cam_tca']),
                'cam_dv': float(row['cam_dv_ms']),
                'num_crossings': int(row['num_degraded_crossings']),
                'num_neighbors': int(row['num_unique_neighbors']),
                'worst_dist': float(row['worst_dist_km']),
                'worst_deg': float(row['worst_degradation_km']),
                'mean_deg': float(row['mean_degradation_km']),
            })

if not cam_effects:
    print("No CAM effects found.")
    exit()

crossings = [c['num_crossings'] for c in cam_effects]
neighbors = [c['num_neighbors'] for c in cam_effects]
worst_dists = [c['worst_dist'] for c in cam_effects]
worst_degs = [c['worst_deg'] for c in cam_effects]

print(f"=== CAM Cascade Analysis ({args.num_sats} satellites) ===")
print(f"Total CAMs with degradation effects: {len(cam_effects)}")
print()
print(f"Degraded crossings per CAM:")
print(f"  Mean:   {np.mean(crossings):.1f}")
print(f"  Median: {np.median(crossings):.0f}")
print(f"  Max:    {max(crossings)}")
print(f"  Total:  {sum(crossings)}")
print()
print(f"Unique neighbors affected per CAM:")
print(f"  Mean:   {np.mean(neighbors):.1f}")
print(f"  Max:    {max(neighbors)}")
print()
print(f"Closest post-CAM approach: {min(worst_dists):.4f} km ({min(worst_dists)*1000:.1f} m)")
print(f"Worst degradation: {max(worst_degs):.4f} km ({max(worst_degs)*1000:.1f} m)")
print()
print(f"Top 15 most disruptive CAMs (by degraded crossings):")
sorted_cams = sorted(cam_effects, key=lambda x: x['num_crossings'], reverse=True)
for c in sorted_cams[:15]:
    print(f"  Sat {c['cam_sat']:>5} (dv={c['cam_dv']:.2f} m/s): "
          f"{c['num_crossings']} crossings, {c['num_neighbors']} neighbors, "
          f"closest {c['worst_dist']*1000:.1f} m")

# Save summary for cross-configuration comparison
summary = f"cascade_summary_{args.num_sats}.csv"
with open(summary, 'w') as f:
    f.write('metric,value\n')
    f.write(f'num_sats,{args.num_sats}\n')
    f.write(f'total_cams_with_effects,{len(cam_effects)}\n')
    f.write(f'mean_crossings_per_cam,{np.mean(crossings):.2f}\n')
    f.write(f'max_crossings_per_cam,{max(crossings)}\n')
    f.write(f'total_degraded_crossings,{sum(crossings)}\n')
    f.write(f'closest_approach_km,{min(worst_dists):.6f}\n')
    f.write(f'worst_degradation_km,{max(worst_degs):.6f}\n')
print(f"\nSummary saved to {summary}")