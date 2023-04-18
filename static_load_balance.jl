using MPI

function main(cartesian_indices, values)
    comm = MPI.COMM_WORLD
    number_of_indices = length(cartesian_indices)
    comm_size = MPI.Comm_size(comm)
    batch_size = ceil(Int, number_of_indices / (Threads.nthreads()*comm_size))
    stride =  MPI.Comm_size(comm)*batch_size
    start_index = MPI.Comm_rank(comm)*batch_size + 1
    println("comm_rank = $(MPI.Comm_rank(comm))")
    println("comm_size = $(comm_size)")
    println("n_threads = $(Threads.nthreads())")
    println("number_of_indices = $(number_of_indices)")
    println("batch_size = $(batch_size)")
    println("stride = $(stride)")
    

    println("start_index = $(start_index)")

    Threads.@sync for batch_index in start_index:stride:number_of_indices
        Threads.@spawn begin
            thread_id = Threads.threadid() 
            for tci_index in batch_index:min(number_of_indices, batch_index + batch_size - 1)
                values[tci_index] = 1.0
            end
        end
    end
    values .= MPI.Allreduce(values, +, comm)
    MPI.Barrier(comm)
end
MPI.Init()
comm = MPI.COMM_WORLD

values = zeros(Float64, (4,4,6))
cartesian_indices = CartesianIndices(values)
main(cartesian_indices, values)

if MPI.Comm_rank(comm) == 0
    for cartesian_index in cartesian_indices
        println("values[$cartesian_index] = $(values[cartesian_index])")
    end
end
MPI.Finalize()
