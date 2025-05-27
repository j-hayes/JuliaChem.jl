
# setup Julia 
# e.g. export PATH=/PATH/TO/Julia/bin/:$PATH
# or module load julia and other modules as needed
# this script assumes that the JuliaChem package is installed in the project path and the LIBINT wrapper is compiled 

module load julia/1.11.4

# export PATH=/pscratch/sd/j/jhayes1/software/julia-1.11.4/bin/:$PATH

export JC_path=/global/cfs/cdirs/m4265/mixed_precision/JuliaChem.jl

export script_path=$JC_path/example_scripts/compile

cd $script_path
export compile_script=$script_path/compile.jl
export JULIACHEM_PRECOMPILE_SCRIPT_PATH=$script_path/run.jl
export JULIACHEM_SYSIMG_PATH=$JC_path/perlmutter_JC_sysimg.so

export input_file_path=$JC_path/example_inputs/compile
export j_project_path=$JC_path/mixed_precision_env_perl


export threads_per_socket=64
export MKL_NUM_THREADS=$threads_per_socket
export OPENBLAS_NUM_THREADS=$threads_per_socket
export JULIA_NUM_THREADS=$threads_per_socket

export inputs_path=$input_file_path
export outputs_path=$script_path


julia --project=$j_project_path --threads=$threads_per_socket --check-bounds=no --math-mode=fast --optimize=3 --inline=yes --compiled-modules=yes $compile_script  &> ./compile.log 
