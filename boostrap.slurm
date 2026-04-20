#!/bin/bash --login
#SBATCH -J testing_2
#SBATCH -p gpuA40GB
#SBATCH -G 1
#SBATCH -n 8
#SBATCH -t 0-2  #Maximum duration is 4 days

module purge
module load apps/binapps/pytorch/2.3.0-311-gpu-cu121

python boostrap.py $SLURM_JOB_ID


