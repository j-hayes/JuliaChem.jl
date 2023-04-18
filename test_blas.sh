#!/bin/bash
#
#SBATCH --job-name=test_blas-1
#SBATCH --output=test_blas-1.log
#SBATCH --error=test_blas-1.err
#
#SBATCH --partition=haswell
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#
export JULIA_NUM_THREADS=16
export OMP_NUM_THREADS=16
#
/export/home/jackson/.julia/bin/mpiexecjl -np 1 julia --check-bounds=no --math-mode=fast --optimize=3 --inline=yes --compiled-modules=yes /export/home/jackson/source/JuliaChem.jl/test_blas_many_threads.jl
