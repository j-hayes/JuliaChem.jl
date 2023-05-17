using Base.Threads
using LinearAlgebra
using TensorOperations
using JuliaChem.Shared.Constants.SCF_Keywords
using JuliaChem.Shared


@inline function calculate_two_center_intgrals(jeri_engine_thread::Vector{T}, basis_sets, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}
    comm = MPI.COMM_WORLD
    aux_basis_function_count = basis_sets.auxillary.norb
    two_center_integrals = zeros(Float64, aux_basis_function_count, aux_basis_function_count)
    auxilliary_basis_shell_count = length(basis_sets.auxillary)
    cartesian_indices = CartesianIndices((auxilliary_basis_shell_count, auxilliary_basis_shell_count))
    n_ranks = MPI.Comm_size(comm)

    max_nbas = max_number_of_basis_functions(basis_sets.auxillary)
    n_threads = Threads.nthreads()
    thead_integral_buffer = [zeros(Float64, max_nbas^2) for i in 1:n_threads]

    if scf_options.load == "sequential"
        calculate_two_center_integrals_sequential!(two_center_integrals, cartesian_indices, jeri_engine_thread[1], thead_integral_buffer[1], basis_sets)
    elseif scf_options.load == "static"
        calculate_two_center_integrals_static!(two_center_integrals, cartesian_indices, jeri_engine_thread, thead_integral_buffer, basis_sets)
    elseif scf_options.load == "dynamic"
        run_two_center_integrals_dynamic!(two_center_integrals, cartesian_indices, jeri_engine_thread, thead_integral_buffer, basis_sets)
    else
        error("integral threading load type: $(scf_options.load) not supported")
    end

    MPI.Barrier(comm)    
    MPI.Allreduce!(two_center_integrals, MPI.SUM, comm)
    MPI.Barrier(comm)   

    return two_center_integrals
end

@inline function calculate_two_center_intgrals_kernel!(two_center_integrals,
    engine,
    cartesian_index,
    basis_sets,
    integral_buffer)
    shell_1_index = cartesian_index[1]
    shell_2_index = cartesian_index[2]

    if shell_2_index > shell_1_index #the top triangle of the symmetric matrix does not need to be calculated
        return
    end

    shell_1 = basis_sets.auxillary.shells[shell_1_index]
    shell_1_basis_count = shell_1.nbas
    bf_1_pos = shell_1.pos

    shell_2 = basis_sets.auxillary.shells[shell_2_index]
    shell_2_basis_count = shell_2.nbas
    bf_2_pos = shell_2.pos

    JERI.compute_two_center_eri_block(engine, integral_buffer, shell_1_index - 1, shell_2_index - 1, shell_1_basis_count, shell_2_basis_count)
    copy_integral_results!(two_center_integrals, integral_buffer, shell_1, shell_2, shell_1_basis_count, shell_2_basis_count)
    axial_normalization_factor(two_center_integrals, shell_1, shell_2, shell_1_basis_count, shell_2_basis_count, bf_1_pos, bf_2_pos)
end

@inline function calculate_two_center_integrals_sequential!(two_center_integrals, cartesian_indices, engine, integral_buffer, basis_sets)
    for cartesian_index in cartesian_indices
        calculate_two_center_intgrals_kernel!(two_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)
    end
end

@inline function calculate_two_center_integrals_static!(two_center_integrals, cartesian_indices, jeri_engine_thread, thead_integral_buffer, basis_sets)
    comm = MPI.COMM_WORLD
    number_of_indices = length(cartesian_indices)
    comm_size = MPI.Comm_size(comm)
    batch_size = ceil(Int, number_of_indices / (Threads.nthreads()*comm_size))

    stride =  comm_size*batch_size
    start_index = MPI.Comm_rank(comm)*batch_size + 1


    @sync for batch_index in start_index:stride:number_of_indices
        Threads.@spawn begin
            thread_id = Threads.threadid()
            for view_index in batch_index:min(number_of_indices, batch_index + batch_size - 1)
                cartesian_index = cartesian_indices[view_index]
                engine = jeri_engine_thread[thread_id]
                integral_buffer = thead_integral_buffer[thread_id]
                calculate_two_center_intgrals_kernel!(two_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)
            end
        end
    end
end

# Dynamic load balancing
# The coordinator rank sends tasks to the worker ranks 
# starting with an initial batch for each rank/thread combination
# then the workers send a message = [0, rank, thread] back to the coordinator rank when they are done with the batch
# the coordinator rank then sends the next batch message = [ij_index] to the worker rank that just finished 
# until all batches are done 
# then the coordinator rank sends a message to all the worker ranks to end the program [ij_index] = -1

# ij_index is the index of the cartesian_indices array. i.e. cartesian_indices[ij_index] =>
# shell_1_index = cartesian_index[1]
# shell_2_index = cartesian_index[2]

@inline function run_two_center_integrals_dynamic!(two_center_integrals, cartesian_indices, jeri_engine_thread, thead_integral_buffer, basis_sets)
    comm = MPI.COMM_WORLD
    n_threads = Threads.nthreads()
    n_indicies = length(cartesian_indices)
    batch_size = size(cartesian_indices, 1)
    rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)

    ## all threads get pre determined first index to process
    task_top_index = [get_top_task_index(n_indicies ,batch_size, n_ranks, n_threads)]

    mutex_mpi_worker = Base.Threads.ReentrantLock()
    @sync for thread in 1:Threads.nthreads()
        Threads.@spawn begin
            threadid = Threads.threadid()
            if rank == 0 && threadid == 1 && n_ranks > 1
                setup_integral_coordinator(task_top_index, batch_size, n_ranks, n_threads, mutex_mpi_worker)
            else
                run_two_center_integrals_worker(two_center_integrals,
                    cartesian_indices,
                    batch_size,
                    jeri_engine_thread,
                    thead_integral_buffer, basis_sets, task_top_index, mutex_mpi_worker, n_indicies)
            end
        end
    end

    cleanup_messages()


end



function run_two_center_integrals_worker(two_center_integrals,
    cartesian_indices,
    batch_size,
    jeri_engine_thread,
    thead_integral_buffer, basis_sets, top_index, mutex_mpi_worker, n_indicies)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    threadid = Threads.threadid()
    n_threads = Threads.nthreads()
    n_ranks = MPI.Comm_size(MPI.COMM_WORLD)

   
    worker_thread_number = get_worker_thread_number(threadid, rank, n_threads, n_ranks)
    ij_index = get_first_task(n_indicies, batch_size, worker_thread_number)
    
    while ij_index > 0
        do_two_center_integral_batch(two_center_integrals,
        ij_index,
        batch_size,
        cartesian_indices,
        jeri_engine_thread[threadid],
        thead_integral_buffer[threadid], basis_sets)
        ij_index = get_next_task(mutex_mpi_worker, top_index, batch_size)
    end
end

@inline function do_two_center_integral_batch(two_center_integrals,
    top_index,
    batch_size,
    cartesian_indices,
    engine,
    integral_buffer, basis_sets)
    for ij in top_index:-1:(max(1, top_index - batch_size))
        shell_index = cartesian_indices[ij]
        calculate_two_center_intgrals_kernel!(two_center_integrals, engine, shell_index, basis_sets, integral_buffer)
    end
end



@inline function copy_integral_results!(two_center_integrals, values, shell_1, shell_2, shell_1_basis_count, shell_2_basis_count)
    temp_index = 1
    for i in shell_1.pos:shell_1.pos+shell_1_basis_count-1
        for j in shell_2.pos:shell_2.pos+shell_2_basis_count-1
            if i >= j # makes sure we don't put any values onto the top triangle which happens for some reason sometimes 
                two_center_integrals[i, j] = values[temp_index]
            else
                two_center_integrals[i, j] = 0.0
            end
            temp_index += 1
        end
    end
end

