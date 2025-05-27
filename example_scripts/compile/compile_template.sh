
# setup Julia 
# e.g. export PATH=/PATH/TO/Julia/bin/:$PATH
# or module load julia and other modules as needed
# this script assumes that the JuliaChem package is installed in the project path and the LIBINT wrapper is compiled 


# export PATH=/pscratch/sd/j/jhayes1/software/julia-1.11.4/bin/:$PATH

export JC_path=<<PATH_TO_JULIACHEM>>

export script_path=<<PATH_TO_THIS_FOLDER>>

cd $script_path
export compile_script=$script_path/compile.jl

export input_file_path=$JC_path/example_inputs/compile

export j_project_path=<<project_path>>


export threads_per_socket=<<NUMBER OF CORES ON EACH SOCKET>>
export MKL_NUM_THREADS=$threads_per_socket
export OPENBLAS_NUM_THREADS=$threads_per_socket
export JULIA_NUM_THREADS=$threads_per_socket

julia --project=$j_project_path --threads=$threads_per_socket --check-bounds=no --math-mode=fast --optimize=3 --inline=yes --compiled-modules=yes $compile_script  &> ./compile.log 
