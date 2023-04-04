#!/bin/bash

#SBATCH --time=1-00:00:00   # walltime (3 days, the maximum)
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=60G   # RAM per CPU core
#SBATCH -J "VoxCeleb"   # job name
#SBATCH --gpus=1


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
# source ~/fsl_groups/grp_mtlab/anaconda3/etc/profile.d/conda.sh
# conda activate opennmt-py
module load miniconda3
source ~/.bashrc
conda activate vox

python3 train.py
