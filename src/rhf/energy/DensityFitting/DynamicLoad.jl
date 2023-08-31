using MPI 

@inline function setup_integral_coordinator(top_index, batch_size, n_ranks, n_threads, mutex_mpi_worker, more_work_tag)
    comm = MPI.COMM_WORLD
    send_msg = [1]
    
    if n_ranks == 1
        return 
    end
    recv_mesg = [0,0]
    rank_thread_has_ended = zeros(Int64, n_ranks, n_threads)
   
    while top_index[1] > 0
        status = MPI.Probe(comm, MPI.Status; source=MPI.ANY_SOURCE, tag=more_work_tag)
        MPI.Recv!(recv_mesg, comm; source=status.source, tag=status.tag)
        send_msg = [get_next_task(mutex_mpi_worker, top_index, batch_size, 1, more_work_tag)]
        if send_msg[1] < 1
            rank_thread_has_ended[recv_mesg[1], recv_mesg[2]] = 1
        end 
        MPI.Send(send_msg, comm; dest=recv_mesg[1], tag=recv_mesg[2])
    end
    
    for rank in 1:n_ranks-1
        for thread in 1:n_threads
            if rank_thread_has_ended[rank, thread] == 1
                continue
            end
            MPI.Send([-1], comm; dest=rank, tag=thread)
        end       
    end
end

@inline function get_next_task(mutex_mpi_worker, top_index, batch_size, threadid, more_work_tag)
    lock(mutex_mpi_worker) do 
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        new_index = 0
        if rank == 0
            new_index = top_index[1]
            top_index[1] -= batch_size + 1
        else
            send_mesg = [MPI.Comm_rank(comm),threadid]
            MPI.Send(send_mesg, comm; dest=0, tag=more_work_tag)
            recv_mesg = [0]
            MPI.Recv!(recv_mesg, comm; source=0, tag=threadid)
            new_index = recv_mesg[1]
        end
        return new_index
    end
end

@inline function cleanup_messages(more_work_tag)
    comm = MPI.COMM_WORLD
    n_ranks = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    if n_ranks == 1
        return 
    end
    #clean up any outstanding requests for work
    ismessage = true
    recv_mesg = [0]
    tag_to_search = MPI.ANY_TAG
    if rank == 0
        tag_to_search = more_work_tag
        recv_mesg = [0,0]    
    end
    
    while ismessage
        ismessage, status = MPI.Iprobe(comm, MPI.Status; source=MPI.ANY_SOURCE, tag=tag_to_search)
        if ismessage
            rreq = MPI.Recv!(recv_mesg, status.source, status.tag, comm) 
        end
    end
end


@inline function get_worker_thread_number(threadid, rank, n_threads, n_ranks)
    worker_thread_number =threadid - 1               
    if n_ranks > 1
        worker_thread_number += rank*n_threads - 1                
    end  
    return worker_thread_number
end

@inline function get_first_task(n_indicies, batch_size, worker_thread_number)
    return n_indicies - (batch_size*worker_thread_number) - worker_thread_number
end

@inline function get_top_task_index(n_indicies ,batch_size, n_ranks, n_threads)
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
    indicies = zeros(Int64, 0)
    i = rank+1
    while i <= number_of_shells
      push!(indicies, i)
      i += comm_size
    end
    return indicies
  end
  
  function static_load_rank_indicies(rank, n_ranks, basis_sets)
    shell_aux_indicies = get_df_static_shell_indices(basis_sets, n_ranks, rank)

    basis_indicies = zeros(Int64, 0)
    for shell_index in shell_aux_indicies
        pos = basis_sets.auxillary[shell_index].pos
        end_index = pos + basis_sets.auxillary[shell_index].nbas-1
        for index in pos:end_index
            push!(basis_indicies, index)
        end
    end
    return shell_aux_indicies, basis_indicies    

  end

  function get_static_gatherv_data(rank, n_ranks, basis_sets, inner_basis_function_length) :: Tuple{Int64, Int64, Int64, Int64, Int64}
    aux_basis_length = length(basis_sets.auxillary)
    begin_index = aux_basis_length÷n_ranks * rank + 1
    end_index = begin_index + aux_basis_length÷n_ranks - 1
    if rank == n_ranks - 1
        end_index = aux_basis_length
    end

    begin_aux_basis_func_index = basis_sets.auxillary[begin_index].pos
    end_aux_basis_func_index = basis_sets.auxillary[end_index].pos + basis_sets.auxillary[end_index].nbas-1
    
    thread_number_of_basis_functions = end_aux_basis_func_index - begin_aux_basis_func_index + 1
    thread_number_of_basis_functions *= inner_basis_function_length

    return begin_index, end_index, begin_aux_basis_func_index, end_aux_basis_func_index, thread_number_of_basis_functions
end

function static_load_thread_index_offset(thread, n_indicies_per_thread)
    return (thread - 1) * n_indicies_per_thread
end

function static_load_thread_shell_to_process_count(thread, nthreads, rank_number_of_shells, n_indicies_per_thread)
    return thread != nthreads ?   n_indicies_per_thread : n_indicies_per_thread + rank_number_of_shells%nthreads
end