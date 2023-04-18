
nprocs=$1
threads=$2

export JULIA_NUM_THREADS=$threads

# mpiexecjl -np $nprocs julia /home/jackson/source/JuliaChem.jl/test_dynamic_tci.jl > ./testoutputs/TCI_Dynamic.log

mpiexecjl -np $nprocs julia /home/jackson/source/JuliaChem.jl/test/density-fitting-test.jl > ./testoutputs/GTFOCK_LessMemory.log

# mpiexecjl -np $nprocs julia /home/jackson/source/JuliaChem.jl/testmpi.jl > ./testoutputs/testmpi.log

# mpiexecjl -np $nprocs julia /home/jackson/source/JuliaChem.jl/learn_mpi.jl > ./testoutputs/learnMPI.log


# mpiexecjl -np $nprocs julia ./static_load_balance.jl  > ./testoutputs/static_load_balance.log