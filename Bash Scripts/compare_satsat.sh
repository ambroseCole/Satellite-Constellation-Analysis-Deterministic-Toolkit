#!/bin/bash
#SBATCH --job-name=compare_satsats
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=satsat_%j.log
#SBATCH --array=0-5

module load python
source venv/SatProject/bin/activate
sats=(1584 3000)
baselines=("results_100.csv" "results_500.csv" "results_1000.csv" "results_1584.csv" "results_3000.csv" "results_5000.csv")
perturbeds=("results_prop_perp_100.csv" "results_prop_perp_500.csv" "results_prop_perp_1000.csv" "results_prop_perp_1584.csv" "results_prop_perp_3000.csv" "results_prop_perp_5000.csv")

i=$SLURM_ARRAY_TASK_ID
python compare_satsat.py --files ${perturbeds[$i]} --num_sats ${sats[$i]}