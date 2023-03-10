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
        status = MPI.Probe(MPI.MPI_ANY_SOURCE, 0, comm) 
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