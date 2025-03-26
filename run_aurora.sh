#!/bin/bash
# mpiexec -n 1 --depth=52 --cpu-bind=list:0-51 /soft/tools/mpi_wrapper_utils/gpu_tile_compact.sh julia --threads=52 ./example_scripts/full-rhf.jl /flare/PorousMatCarbon/jacksonjhayes/source/JuliaChem.jl/example_inputs/density_fitting/water_density_fitted.json &> ./density_fitting_cpu.log
PATH=/lus/flare/projects/PorousMatCarbon/jacksonjhayes/languages/julia-1.10.8/bin:$PATH

# mpiexec -n 1 --depth=52 --cpu-bind=list:0-51 /soft/tools/mpi_wrapper_utils/gpu_tile_compact.sh julia --threads=52 ./example_scripts/full-rhf.jl /flare/PorousMatCarbon/jacksonjhayes/source/JuliaChem.jl/example_inputs/density_fitting/water_density_fitted_gpu.json &> ./density_fitting_denseGPU_refactor.log
# export input_file=/lus/flare/projects/PorousMatCarbon/jacksonjhayes/source/JuliaChem.jl/example_inputs/density_fitting/C20H42_dfGPU.json
# export input_file=/flare/PorousMatCarbon/jacksonjhayes/source/JuliaChem.jl/example_inputs/density_fitting/water_density_fitted_gpu.json
mpiexec -n 1 --depth=52 --cpu-bind=list:0-51 /soft/tools/mpi_wrapper_utils/gpu_tile_compact.sh julia --threads=52 ./example_scripts/full-rhf.jl $input_file &> ./screened_water_noK.log