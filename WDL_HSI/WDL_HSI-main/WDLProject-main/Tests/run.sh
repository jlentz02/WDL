#!/bin/sh
#SBATCH -J fixed_data 
#SBATCH --time=00-48:00:00 
#SBATCH -p batch #running on mpi partition
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=48g
#SBATCH --output=MyJob.%j.%N.out #output file
#SBATCH --error=MyJob.%j.%N.out #Error file
#SBATCH --mail-type=END
#SBATCH --mail-user=*


module load anaconda/*
source activate /* #Path to conda environment goes here
python3 salinas_run.py "--n_atoms=$1" "--geom=$2" "--recip=$3"


