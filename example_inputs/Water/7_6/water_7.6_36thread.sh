#!/bin/bash
#
#SBATCH --job-name=w7_6_36thread
#SBATCH --output=w7_6_36thread.log
#SBATCH --error=w7_6_36thread.err
#
#SBATCH --partition=haswell
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#
export JULIA_NUM_THREADS=36
export OPENBLAS_NUM_THREADS=1
#
orterun --mca btl vader,self,tcp -np 1 -map-by ppr:1:node --bind-to none --report-bindings julia --check-bounds=no --math-mode=fast --optimize=3 --inline=yes --compiled-modules=yes ../../../example_scripts/minimal-rhf-benchmark.jl water_7.6.json
