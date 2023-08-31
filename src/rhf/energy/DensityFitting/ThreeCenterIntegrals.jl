using Base.Threads
using LinearAlgebra
using TensorOperations
using JuliaChem.Shared.Constants.SCF_Keywords
using JuliaChem.Shared

three_center_integral_tag = 3000


@inline function calculate_three_center_integrals(jeri_engine_thread, basis_sets::CalculationBasisSets, scf_options::SCFOptions)
    aux_basis_function_count = basis_sets.auxillary.norb
    basis_function_count = basis_sets.primary.norb
    three_center_integrals = zeros(Float64, (basis_function_count, basis_function_count, aux_basis_function_count))
    auxilliary_basis_shell_count = length(basis_sets.auxillary)
    basis_shell_count = length(basis_sets.primary)

    cartesian_indices = CartesianIndices((auxilliary_basis_shell_count, basis_shell_count, basis_shell_count))
    n_threads = Threads.nthreads()

    max_primary_nbas = max_number_of_basis_functions(basis_sets.primary)
    max_aux_nbas = max_number_of_basis_functions(basis_sets.auxillary)
    thead_integral_buffer = [zeros(Float64, max_primary_nbas^2 * max_aux_nbas) for thread in 1:n_threads]


    if scf_options.load == "sequential"
        calculate_three_center_integrals_sequential!(three_center_integrals, thead_integral_buffer[1], cartesian_indices, jeri_engine_thread[1], basis_sets)
    elseif scf_options.load == "static"
        calculate_three_center_integrals_static(three_center_integrals, jeri_engine_thread, basis_sets, thead_integral_buffer)
    elseif scf_options.load == "dynamic"
        calculate_three_center_integrals_dynamic!(three_center_integrals, cartesian_indices, jeri_engine_thread, basis_sets, thead_integral_buffer)
    else
        error("integral threading load type: $(scf_options.load) not supported")
    end
    return three_center_integrals
end

@inline function get_indexes_eri_block(cartesian_index :: CartesianIndex, basis_sets :: CalculationBasisSets)
    s1 = cartesian_index[1]
    s2 = cartesian_index[2]
    s3 = cartesian_index[3]

    shell_1 = basis_sets.auxillary.shells[s1]
    shell_1_nbasis = shell_1.nbas
    bf_1_pos = shell_1.pos

    shell_2 = basis_sets.primary.shells[s2]
    shell_2_nbasis = shell_2.nbas
    bf_2_pos = shell_2.pos
    n12 = shell_1_nbasis * shell_2_nbasis

    shell_3 = basis_sets.primary.shells[s3]
    shell_3_nbasis = shell_3.nbas
    bf_3_pos = shell_3.pos

    number_of_integrals = n12 * shell_3_nbasis
    return s1, s2, s3, shell_1, shell_2, shell_3, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis, bf_1_pos, bf_2_pos, bf_3_pos, number_of_integrals #todo make this a struct
end

@inline function calculate_three_center_integrals_kernel!(three_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)
    s1, s2, s3,
    shell_1, shell_2, shell_3,
    shell_1_nbasis, shell_2_nbasis, shell_3_nbasis, 
    bf_1_pos, bf_2_pos, bf_3_pos, 
    number_of_integrals = get_indexes_eri_block(cartesian_index, basis_sets)
    
    JERI.compute_eri_block_df(engine, integral_buffer, s1, s2, s3, number_of_integrals, 0)
    copy_integral_result!(three_center_integrals, integral_buffer, bf_1_pos, bf_2_pos, bf_3_pos, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis)
    axial_normalization_factor(three_center_integrals, shell_1, shell_2, shell_3, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis, bf_1_pos, bf_2_pos, bf_3_pos)

end

@inline function recieve_messages(three_center_integrals, basis_sets)
    comm = MPI.COMM_WORLD
    n_ranks_still_working = MPI.Comm_size(comm) - 1 # this process is not responsible for the worker threads on rank 0
    communication_buffer = Dict{CartesianIndex, Array{Float64, 1}}()
    while n_ranks_still_working > 0 
        status = MPI.Probe(-2, more_work_tag, comm)
        rreq = MPI.Recv!(communication_buffer, status.source, status.tag, comm)
        if length(keys(communication_buffer)) > 0
            process_integral_message(communication_buffer, three_center_integrals, basis_sets)
        else
            n_ranks_still_working -= 1
        end
    end
end



@inline function process_integral_message(communication_buffer, three_center_integrals, basis_sets)
    for key in keys(communication_buffer, communication_buffer_lock)
        
        integral_buffer = communication_buffer[key]

        s1, s2, s3, 
        shell_1, shell_2, shell_3,
        shell_1_nbasis, shell_2_nbasis, shell_3_nbasis, 
        bf_1_pos, bf_2_pos, bf_3_pos, 
        number_of_integrals = get_indexes_eri_block(key, basis_sets)

        copy_integral_result!(three_center_integrals, integral_buffer, bf_1_pos, bf_2_pos, bf_3_pos, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis)
        axial_normalization_factor(three_center_integrals, shell_1, shell_2, shell_3, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis, bf_1_pos, bf_2_pos, bf_3_pos)
    end
end


@inline function calculate_three_center_integrals_sequential!(three_center_integrals, integral_buffer, cartesian_indices, engine, basis_sets)
    for cartesian_index in cartesian_indices
        calculate_three_center_integrals_kernel!(three_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)
    end
end


@inline function calculate_three_center_integrals_static(three_center_integrals, jeri_engine_thread, basis_sets, thead_integral_buffer)
    comm = MPI.COMM_WORLD
    n_ranks = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    nthreads = Threads.nthreads()
    basis_length = length(basis_sets.primary)
    gatherv_data = [get_static_gatherv_data(x, n_ranks, basis_sets, (basis_sets.primary.norb)^2) for x in 0:n_ranks-1]
    indicies_per_rank = [x[5] for x in gatherv_data]
    begin_index = gatherv_data[rank+1][1]
    end_index = gatherv_data[rank+1][2]
    begin_aux_basis_func_index =  gatherv_data[rank+1][3]
    end_aux_basis_func_index = gatherv_data[rank+1][4]
    

    n_indicies_per_thread = (end_index - begin_index + 1)÷nthreads    

    Threads.@sync for thread in 1:nthreads
        Threads.@spawn begin                

            thread_begin_index = begin_index + (thread - 1) * n_indicies_per_thread
            thread_end_index = thread_begin_index + n_indicies_per_thread - 1
            if thread == nthreads
                thread_end_index = end_index
            end

            for aux_index in thread_begin_index:thread_end_index
                for μ in 1:basis_length
                    for ν in 1:basis_length
                        cartesian_index = CartesianIndex(aux_index, μ, ν)
                        engine = jeri_engine_thread[thread]
                        integral_buffer = thead_integral_buffer[thread]
                        calculate_three_center_integrals_kernel!(three_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)
                    end
                end    
            end
        end
    end
    if n_ranks > 1
        three_center_integral_buff = MPI.VBuffer(three_center_integrals, indicies_per_rank) # setup the root with the actual buffer to recieve data
        # MPI.Allreduce!(three_center_integrals, MPI.SUM, comm)
        MPI.Allgatherv!(three_center_integrals[:,:,begin_aux_basis_func_index:end_aux_basis_func_index], three_center_integral_buff, comm)
    end
end




@inline function calculate_three_center_integrals_dynamic!(three_center_integrals, cartesian_indices, 
    jeri_engine_thread, basis_sets, thead_integral_buffer)
    comm = MPI.COMM_WORLD
    n_threads = Threads.nthreads()
    n_indicies = length(cartesian_indices)
    batch_size = size(cartesian_indices, 1)

    rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)

    # all threads get pre determined first index to process this is the next lowest index for processing
    task_top_index = [get_top_task_index(n_indicies, batch_size, n_ranks, n_threads)]

    mutex_mpi_worker = Base.Threads.ReentrantLock()
    @sync for thread in 1:Threads.nthreads()
        Threads.@spawn begin
            if rank == 0 && thread == 1 && n_ranks > 1
                setup_integral_coordinator(task_top_index, batch_size, n_ranks, n_threads, mutex_mpi_worker, three_center_integral_tag)
            else
                run_three_center_integrals_worker(three_center_integrals,
                cartesian_indices,
                batch_size,
                jeri_engine_thread,
                thead_integral_buffer, basis_sets, task_top_index, mutex_mpi_worker, n_indicies, thread)
            end
        end
    end
   
    if n_ranks > 1
        MPI.Allreduce!(three_center_integrals, MPI.SUM, comm)
        cleanup_messages(three_center_integral_tag) #todo figure out why there are extra messages and remove this
    end
end


function run_three_center_integrals_worker(three_center_integrals,
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
        ijk_index = get_first_task(n_indicies, batch_size, worker_thread_number)
    unlock(mutex_mpi_worker)
    while ijk_index > 0
        do_three_center_integral_batch!(three_center_integrals,
        ijk_index,
        batch_size,
        cartesian_indices,
        jeri_engine_thread[thread],
        thead_integral_buffer[thread], basis_sets)
        ijk_index = get_next_task(mutex_mpi_worker, top_index, batch_size, thread, three_center_integral_tag)
    end
end


@inline function do_three_center_integral_batch!(three_center_integrals,
    top_index,
    batch_size,
    cartesian_indices,
    engine,
    integral_buffer, basis_sets)
    for ij in top_index:-1:(max(1, top_index - batch_size))
        shell_index = cartesian_indices[ij]
        calculate_three_center_integrals_kernel!(three_center_integrals, engine, shell_index, basis_sets, integral_buffer)
    end
end

@inline function copy_integral_result!(three_center_integrals, values, bf_1_pos, bf_2_pos, bf_3_pos, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis)
    values_index = 1
    for i in bf_1_pos:bf_1_pos+shell_1_nbasis-1
        for j in bf_2_pos:bf_2_pos+shell_2_nbasis-1
            for k in bf_3_pos:bf_3_pos+shell_3_nbasis-1
                three_center_integrals[j, k, i] = values[values_index]
                values_index += 1
            end
        end
    end
end