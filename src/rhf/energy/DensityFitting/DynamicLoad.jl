using MPI 
more_work_tag = 100

@inline function setup_integral_coordinator(top_index, batch_size, n_ranks, n_threads, mutex_mpi_worker)
    comm = MPI.COMM_WORLD
    send_msg = [1]
    
    if n_ranks == 1
        return 
    end
    recv_mesg = [0,0,0]
    while top_index[1] > 0
        status = MPI.Probe(-2, more_work_tag, comm)
        rreq = MPI.Recv!(recv_mesg, status.source, status.tag, comm)
        send_msg = [get_next_task(mutex_mpi_worker, top_index, batch_size, top_index[1])]
        MPI.Send(send_msg, recv_mesg[1], recv_mesg[2], comm)
    end

    for rank in 1:n_ranks-1
        for thread in 1:n_threads
            sreq = MPI.Send([-1], rank, thread, comm)
        end       
    end
end

function get_next_task(index_lock, top_index, batch_size, current_index)
    lock(index_lock) do 
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        new_index = 0
        if rank == 0
            new_index = top_index[1]
            top_index[1] -= batch_size + 1
        else
            send_mesg = [MPI.Comm_rank(comm), Threads.threadid(), current_index]
            MPI.Send(send_mesg, 0, more_work_tag, comm)
            recv_mesg = [0]
            MPI.Recv!(recv_mesg, 0, Threads.threadid(), comm)
            if recv_mesg[1] < 1
                println("recieved end message for rank: $(send_mesg[1]) thread: $(send_mesg[2])")
            end
            new_index = recv_mesg[1]

        end
        return new_index
    end
end

function cleanup_messages()
    comm = MPI.COMM_WORLD
    n_ranks = MPI.Comm_size(comm)
    if n_ranks == 1
        return 
    end
    #clean up any outstanding requests for work
    ismessage = true
    recv_mesg = [0]
        if rank == 0
        recv_mesg = [0,0,0]    
    end
    while ismessage
        ismessage, status = MPI.Iprobe(-2, -1, comm)
        if ismessage
            rreq = MPI.Recv!(recv_mesg, status.source, status.tag, comm) 
        end
    end
end


function get_worker_thread_number(threadid, rank, n_threads, n_ranks)
    worker_thread_number = 0                
    if n_ranks > 1
        worker_thread_number = threadid + (rank*n_threads-2)                
    else
        worker_thread_number = threadid + (rank*n_threads-1)    
    end   
    return worker_thread_number
end

function get_first_task(n_indicies, batch_size, worker_thread_number)
    return n_indicies - (batch_size*worker_thread_number) - worker_thread_number
end

function get_top_task_index(n_indicies ,batch_size, n_ranks, n_threads)
    task_top_index = n_indicies
    for r in 0:n_ranks-1 
        for t in 1:n_threads
            if r == 0 && t == 1 && n_ranks > 1
                continue
            end
            task_top_index -= batch_size + 1
        end
    end
    return task_top_index
    # task_top_index[1] -= (batch_size + 1)*n_ranks*n_threads
    # if n_ranks > 1 #account for coordinator 
    #     task_top_index[1] += batch_size + 1
    # end
end



# move these to a static load file
@inline function get_df_static_basis_indices(basis_sets, comm_size, rank)
    number_of_shells = length(basis_sets.auxillary)
    indicies = []
    i = rank+1
    while i <= number_of_shells
      pos = basis_sets.auxillary[i].pos
      end_index = pos + basis_sets.auxillary[i].nbas-1
      for index in pos:end_index
        push!(indicies, index)
      end
      i += comm_size
    end
    return indicies
  end
  
  
  @inline function get_df_static_shell_indices(basis_sets, comm_size, rank)
    number_of_shells = length(basis_sets.auxillary)
    indicies = []
    i = rank+1
    while i <= number_of_shells
      push!(indicies, i)
      i += comm_size
    end
    return indicies
  end
  
