#!/bin/bash
#SBATCH --job-name=sat_debris_cam
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=50:00:00
#SBATCH --output=cam_logs/cam_%A_%a.log
#SBATCH --array=0-49

module load python
source venv/SatProject/bin/activate
export PYTHONUNBUFFERED=1

sats=(1584 3000)
planes=(72 75)
windows=10  # 30 days / 3 days per window

# Decode array index: constellation + window
config_idx=$(( SLURM_ARRAY_TASK_ID / windows ))
window_idx=$(( SLURM_ARRAY_TASK_ID % windows ))

num_sats=${sats[$config_idx]}
num_planes=${planes[$config_idx]}
start_day=$(( window_idx * 3 ))
end_day=$(( start_day + 3 ))

echo "Config: ${num_sats} sats, ${num_planes} planes, days ${start_day}-${end_day}"

python sat_debris_cam.py \
    --num_sats $num_sats \
    --num_planes $num_planes \
    --start_day $start_day \
    --end_day $end_day \
    --output results_cam_${num_sats}.csv