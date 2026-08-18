#!/bin/bash
#SBATCH --job-name=propagate_perturbed
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=prop_perp_%j.log
#SBATCH --array=0-5

module load python
source venv/SatProject/bin/activate
sats=(100 500 1000 1584 3000)
planes=(10 25 50 72 75)
cam_files=("results_0pol_cam_100.csv" "results_0pol_cam_500.csv" "results_0pol_cam_1000.csv" "results_F10_cam_1584.csv" "results_F10_cam_3000.csv" "results_F10_cam_5000.csv")
baselines=("results_100.csv" "results_500.csv" "results_1000.csv" "results_1584.csv" "results_3000.csv" "results_5000.csv")

i=$SLURM_ARRAY_TASK_ID
python propagate_perturbed.py --cam_file ${cam_files[$i]} --num_sats ${sats[$i]} --baseline_file ${baselines[$i]} --num_planes ${planes[$i]} --output results_prop_perp_${sats[$i]}.csv