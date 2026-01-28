#!/bin/bash
#SBATCH -J com_poisson
#SBATCH --account=overcap
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --exclude=jill,johnny5,bb8,calculon,irona
#SBATCH --qos="short" 
#SBATCH -t 30
#SBATCH -o Report-%A-%a.out
#SBATCH --array=100-299
cd $SLURM_SUBMIT_DIR

date

srun python main.py --idx $SLURM_ARRAY_TASK_ID --mode both

date
