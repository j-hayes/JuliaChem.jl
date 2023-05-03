
threads=$1
outputpath=$2

export JULIA_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

#use sysimg
julia -J"../SYSIMG/JuliaChem.so" --check-bounds=no --math-mode=fast --optimize=3 --inline=yes --compiled-modules=yes ./density-fitting-vs-rhf.jl > $outputpath

# use without sysimg
# julia --check-bounds=no --math-mode=fast --optimize=3 --inline=yes --compiled-modules=yes ./density-fitting-vs-rhf.jl > $outputpath
