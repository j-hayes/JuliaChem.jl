#!/bin/bash
#
#SBATCH --job-name=juliachem-runtest
#SBATCH --output=juliachem-runtest.log
#SBATCH --error=juliachem-runtest.err
#
#SBATCH --partition=haswell
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#
export JULIA_NUM_THREADS=1
#
mpirun -np 1 julia --check-bounds=no --math-mode=fast --optimize=3 --inline=yes --compiled-modules=yes -E 'using Pkg; using JuliaChem; Pkg.test("JuliaChem")'
