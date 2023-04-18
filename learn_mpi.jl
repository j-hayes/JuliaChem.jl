using MPI
using LinearAlgebra
using Random

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    number_of_threads = Threads.nthreads()
    numbers_to_process = 1000
    batch_size = 2
    integrals = zeros(numbers_to_process)
    println("batch_size: $(batch_size)\n")
    setup_coordinator(numbers_to_process, batch_size)
    setup_workers(batch_size, integrals)
    MPI.Barrier(comm)
    if rank == 0
        for i in 1:numbers_to_process
            println("$i: $(integrals[i])")
        end
        println("done")
    end
    MPI.Finalize()
end

function setup_coordinator(top_index, batch_size)
    comm = MPI.COMM_WORLD
    n_threads = Threads.nthreads()
    if MPI.Comm_rank(comm) != 0
        return
    end
    
    number_of_ranks = MPI.Comm_size(comm)
    for rank in 1:number_of_ranks-1
        for thread in 1:n_threads
          sreq = MPI.Isend([ top_index ], rank, thread, comm)
          top_index -= batch_size + 1
        end
    end
    recv_mesg = [ 0,0,0 ] 
    while top_index > 0
        status = MPI.Probe(MPI.MPI_ANY_SOURCE, 0, comm) 
        rreq = MPI.Recv!(recv_mesg, status.source, status.tag, comm)
        sreq = MPI.Isend([ top_index ], recv_mesg[2], recv_mesg[3], comm)
        top_index -= batch_size + 1
    end
    println("sent all messages")

    for rank in 1:number_of_ranks-1
        for thread in 1:n_threads
          sreq = MPI.Isend([ -1 ], rank, thread, comm)
        end
    end

end

function setup_workers(batch_size, integrals)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    integral_workspace = zeros(size(integrals))
    if rank  != 0
        mutex_mpi_worker = Base.Threads.ReentrantLock()
        #== execute kernel ==# 
        @sync for thread in 1:Threads.nthreads()
            Threads.@spawn begin 
                #== initial set up ==#
                recv_mesg = [ 0 ] 
                send_mesg = [ rank ]   
                #== complete first task ==#
                lock(mutex_mpi_worker)
                    status = MPI.Probe(0, thread, comm)
                    rreq = MPI.Recv!(recv_mesg, status.source, status.tag, comm)
                    ij_index = recv_mesg[1]
                unlock(mutex_mpi_worker)  
                while ij_index >= 1 
                    sleep(.01)  

                    for i in max(1, ij_index-batch_size):ij_index
                        integral_workspace[i] = 100*rank+thread
                    end
                    ij_index = get_next_batch(mutex_mpi_worker, send_mesg, recv_mesg, comm, thread)
                end
                println("done on rank $(rank) thread $(thread)")
            end 
        end
    end
    MPI.Barrier(comm)
    integrals .= MPI.Allreduce(integral_workspace, MPI.SUM, comm)
    MPI.Barrier(comm)

end

function get_next_batch(mutex_mpi_worker, send_mesg, recv_mesg, comm, thread)
    lock(mutex_mpi_worker)
      send_mesg = [ 0 , MPI.Comm_rank(comm), thread ]   
      status = MPI.Isend(send_mesg, 0, 0, comm)
      status = MPI.Probe(0, thread, comm)
      rreq = MPI.Recv!(recv_mesg, status.source, status.tag, comm)
      ij_index = recv_mesg[1]
    unlock(mutex_mpi_worker)
    return ij_index
end

main()