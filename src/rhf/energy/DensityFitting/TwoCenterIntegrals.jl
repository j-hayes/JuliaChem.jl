using Base.Threads
using LinearAlgebra
using TensorOperations
using JuliaChem.Shared.Constants.SCF_Keywords
using JuliaChem.Shared

two_center_integral_tag = 2000

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


    # for index in CartesianIndices(two_center_integrals)
    #     println("TCI[$(index[1]), $(index[2])] = $(two_center_integrals[index])")
    # end
    
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
    rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)
    comm_size = MPI.Comm_size(comm)
    nthreads = Threads.nthreads()
    aux_basis_length = length(basis_sets.auxillary)
    # number_of_indices = length(cartesian_indices)
    # batch_size = ceil(Int, number_of_indices / (Threads.nthreads()*comm_size))
    # stride =  comm_size*batch_size
    # start_index = MPI.Comm_rank(comm)*batch_size + 1

    # @sync for thread in 1:nthreads
    #     Threads.@spawn begin
    #         batch_index = start_index + (thread - 1)*stride
    #         for view_index in batch_index:min(number_of_indices, batch_index + batch_size - 1)
    #             cartesian_index = cartesian_indices[view_index]
    #             engine = jeri_engine_thread[thread]
    #             integral_buffer = thead_integral_buffer[thread]
    #             calculate_two_center_intgrals_kernel!(two_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)
    #         end
    #     end
    # end

    gatherv_data = [get_static_gatherv_data(x, n_ranks, basis_sets, basis_sets.auxillary.norb) for x in 0:n_ranks-1]
    indicies_per_rank = [x[5] for x in gatherv_data]
    begin_index = gatherv_data[rank+1][1]
    end_index = gatherv_data[rank+1][2]
    begin_aux_basis_func_index =  gatherv_data[rank+1][3]
    end_aux_basis_func_index = gatherv_data[rank+1][4]
   
    n_indicies_per_thread = (end_index - begin_index + 1)÷nthreads   





    println("rank $(rank) begin: $(begin_index) end: $(end_index) aux begin: $(begin_aux_basis_func_index) aux end: $(end_aux_basis_func_index)")

    println("aux shell count = $(length(basis_sets.auxillary))")
    Threads.@sync for thread in 1:nthreads
        Threads.@spawn begin     
            thread_begin_index = begin_index + (thread - 1) * n_indicies_per_thread
            thread_end_index = thread_begin_index + n_indicies_per_thread - 1
            if thread == nthreads
                thread_end_index = end_index
            end
            # println("thread: $(thread) begin: $(thread_begin_index) end: $(thread_end_index)")

            for B in thread_begin_index:thread_end_index
                for A in 1:aux_basis_length
                    cartesian_index = CartesianIndex(A, B)
                    engine = jeri_engine_thread[thread]
                    integral_buffer = thead_integral_buffer[thread]
                    calculate_two_center_intgrals_kernel!(two_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)
                end
            end    
        end
    end

    if n_ranks > 1
        two_center_integral_buff = MPI.VBuffer(two_center_integral_buff, indicies_per_rank) # setup the root with the actual buffer to recieve data
        MPI.Allgatherv!(two_center_integrals[:,:,begin_aux_basis_func_index:end_aux_basis_func_index], two_center_integral_buff, comm)
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
            if rank == 0 && thread == 1 && n_ranks > 1
                setup_integral_coordinator(task_top_index, batch_size, n_ranks, n_threads, mutex_mpi_worker, two_center_integral_tag)
            else
                run_two_center_integrals_worker(two_center_integrals,
                    cartesian_indices,
                    batch_size,
                    jeri_engine_thread,
                    thead_integral_buffer, basis_sets, task_top_index, mutex_mpi_worker, n_indicies, thread)
            end
        end
    end


    if n_ranks > 1
        MPI.Allreduce!(two_center_integrals, MPI.SUM, comm)
        cleanup_messages(two_center_integral_tag) #todo figure out why there are extra messages and remove this
    end
end



@inline function run_two_center_integrals_worker(two_center_integrals,
    cartesian_indices,
    batch_size,
    jeri_engine_thread,
    thead_integral_buffer, basis_sets, top_index, mutex_mpi_worker, n_indicies, thread)

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n_threads = Threads.nthreads()
    n_ranks = MPI.Comm_size(comm)

    
    lock(mutex_mpi_worker)
        worker_thread_number = get_worker_thread_number(thread, rank, n_threads, n_ranks)
        ij_index = get_first_task(n_indicies, batch_size, worker_thread_number)
    unlock(mutex_mpi_worker)

    while ij_index > 0
        do_two_center_integral_batch(two_center_integrals,
        ij_index,
        batch_size,
        cartesian_indices,
        jeri_engine_thread[thread],
        thead_integral_buffer[thread], basis_sets)
        ij_index = get_next_task(mutex_mpi_worker, top_index, batch_size, thread, two_center_integral_tag)
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

