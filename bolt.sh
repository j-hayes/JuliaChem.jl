#!/bin/bash
#
#SBATCH --job-name=s22_benchmark
#SBATCH --output=s22_benchmark-2-36.log
#SBATCH --error=s22_benchmark-2-36.err
#
#SBATCH --partition=haswell
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#
export JULIA_NUM_THREADS=36
export OMP_NUM_THREADS=36
#
/export/home/jackson/.julia/bin/mpiexecjl -np 1 julia --check-bounds=no --math-mode=fast --optimize=3 --inline=yes --compiled-modules=yes density-fitting-test.jl
