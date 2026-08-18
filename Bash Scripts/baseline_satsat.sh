#!/bin/bash
#SBATCH --job-name=baseline_cam
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=train_%j.log
#SBATCH --array=0-5

module load python
source venv/SatProject/bin/activate
sats=(100 500 1000 1584 3000)
planes=(10 25 50 72 75)

i=$SLURM_ARRAY_TASK_ID
python baseline_satsat.py --num_sats ${sats[$i]} --num_planes ${planes[$i]} --output results_${sats[$i]}.csv