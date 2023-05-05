using MPI 
more_work_tag = 100

@inline function setup_integral_coordinator(top_index, batch_size, n_ranks, n_threads, mutex_mpi_worker)
    comm = MPI.COMM_WORLD
    send_msg = [1]
    index_to_send = 0
    
    if n_ranks == 1
        return 
    end

    for rank in 1:n_ranks-1
        for thread in 1:n_threads
            index_to_send = get_next_task(mutex_mpi_worker, top_index, batch_size)
            MPI.Send([index_to_send], rank, thread, comm)
        end
    end

    while top_index[1] > 0
        status = MPI.Probe(-2, more_work_tag, comm)
        recv_mesg = [0,0]
        rreq = MPI.Recv!(recv_mesg, status.source, more_work_tag, comm)
        send_msg = [get_next_task(mutex_mpi_worker, top_index, batch_size)]
        if send_msg[1] > 0
            MPI.Send(send_msg, recv_mesg[1], recv_mesg[2], comm)
        end
    end

    for rank in 1:n_ranks-1
        for thread in 1:n_threads
            MPI.Send([-1], rank, thread, comm)
        end
    end
end

@inline function get_next_task(mutex_mpi_worker, top_index, batch_size)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    new_index = 0
    if rank == 0
        lock(mutex_mpi_worker)
        new_index = top_index[1]
        top_index[1] -= batch_size + 1
        unlock(mutex_mpi_worker)
    else
        lock(mutex_mpi_worker)
        send_mesg = [MPI.Comm_rank(comm), Threads.threadid()]
        MPI.Send(send_mesg, 0, more_work_tag, comm)
        recv_mesg = [0]
        MPI.Recv!(recv_mesg, 0, Threads.threadid(), comm)
        new_index = recv_mesg[1]
        unlock(mutex_mpi_worker)
    end

    return new_index
end
    

@inline function setup_integral_coordinator(task_top_index, batch_size, n_ranks, n_threads)
    task_top_index = send_initial_tasks_to_workers!(task_top_index, batch_size, n_ranks, n_threads)
    send_integral_tasks_dynamic(task_top_index, batch_size)
    send_end_signals(n_ranks, n_threads)
end

@inline function send_initial_tasks_to_workers!(task_top_index, batch_size, n_ranks, n_threads)
    for rank in 1:n_ranks-1
        for thread in 1:n_threads
            sreq = MPI.Isend([task_top_index], rank, thread, MPI.COMM_WORLD)
            task_top_index -= batch_size + 1
        end
    end
    return task_top_index
end


@inline function send_integral_tasks_dynamic(task_top_index, batch_size)
    comm = MPI.COMM_WORLD
    recv_mesg = [0,0,0] # message type, rank, thread
    while task_top_index > 0
        status = MPI.Probe(-2, 0, comm) # -2 = MPI.MPI_ANY_SOURCE
        rreq = MPI.Recv!(recv_mesg, status.source, status.tag, comm)
        sreq = MPI.Isend([ task_top_index ], recv_mesg[2], recv_mesg[3], comm)
        task_top_index -= batch_size + 1
    end
end

@inline function send_end_signals(n_ranks, n_threads)
    for rank in 1:(n_ranks-1)
        for thread in 1:n_threads
            sreq = MPI.Isend([-1], rank, thread, MPI.COMM_WORLD)
        end
    end
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
  
