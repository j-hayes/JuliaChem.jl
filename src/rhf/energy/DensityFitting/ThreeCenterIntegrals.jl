using Base.Threads
using LinearAlgebra
using TensorOperations
using JuliaChem.Shared.Constants.SCF_Keywords
using JuliaChem.Shared

three_center_integral_tag = 3000


@inline function calculate_three_center_integrals(jeri_engine_thread, basis_sets::CalculationBasisSets, scf_options::SCFOptions)
    comm = MPI.COMM_WORLD

    aux_basis_function_count = basis_sets.auxillary.norb
    basis_function_count = basis_sets.primary.norb
    three_center_integrals = zeros(Float64, (basis_function_count, basis_function_count, aux_basis_function_count))
    auxilliary_basis_shell_count = length(basis_sets.auxillary)
    basis_shell_count = length(basis_sets.primary)

    cartesian_indices = CartesianIndices((auxilliary_basis_shell_count, basis_shell_count, basis_shell_count))
    number_of_indices = length(cartesian_indices)
    n_threads = Threads.nthreads()
    n_ranks = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    batch_size = ceil(Int, number_of_indices / n_threads)

    max_primary_nbas = max_number_of_basis_functions(basis_sets.primary)
    max_aux_nbas = max_number_of_basis_functions(basis_sets.auxillary)
    thead_integral_buffer = [zeros(Float64, max_primary_nbas^2 * max_aux_nbas) for thread in 1:n_threads]

    communication_buffer = Dict{CartesianIndex, Array{Float64, 1}}()
    communication_buffer_lock = Base.Threads.ReentrantLock()

    if scf_options.load == "sequential"
        calculate_three_center_integrals_sequential!(three_center_integrals, thead_integral_buffer[1], cartesian_indices, jeri_engine_thread[1], basis_sets, communication_buffer, communication_buffer_lock)
    elseif scf_options.load == "static"
        calculate_three_center_integrals_static(three_center_integrals, cartesian_indices, jeri_engine_thread, basis_sets, thead_integral_buffer, communication_buffer, communication_buffer_lock)
    elseif scf_options.load == "dynamic"
        calculate_three_center_integrals_dynamic!(three_center_integrals, cartesian_indices, jeri_engine_thread, basis_sets, thead_integral_buffer, communication_buffer, communication_buffer_lock)
    else
        error("integral threading load type: $(scf_options.load) not supported")
    end
    if rank > 0
        if length(keys(communication_buffer)) > 0
            MPI.Send(communication_buffer, 0, three_center_integral_tag, comm)
            communication_buffer = Dict{CartesianIndex, Array{Float64, 1}}()
        end
        MPI.Send(communication_buffer, 0, three_center_integral_tag, comm)
    end
    # if n_ranks > 1
    #     MPI.Barrier(comm)    
    #     MPI.Allreduce!(three_center_integrals, MPI.SUM, comm)
    #     MPI.Barrier(comm)   
    # end


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

@inline function calculate_three_center_integrals_kernel!(three_center_integrals, engine, cartesian_index, basis_sets, integral_buffer,
    communication_buffer, communication_buffer_lock)
    comm = MPI.COMM_WORLD

    s1, s2, s3,
    shell_1, shell_2, shell_3,
    shell_1_nbasis, shell_2_nbasis, shell_3_nbasis, 
    bf_1_pos, bf_2_pos, bf_3_pos, 
    number_of_integrals = get_indexes_eri_block(cartesian_index, basis_sets)
    
    JERI.compute_eri_block_df(engine, integral_buffer, s1, s2, s3, number_of_integrals, 0)

    if MPI.Comm_rank(comm) > 0
        lock(communication_buffer_lock) 
        communication_buffer[cartesian_index] = deepcopy(integral_buffer[1:number_of_integrals])    
        if length(keys(communication_buffer, communication_buffer_lock)) > 10
            MPI.ISend(communication_buffer, 0, three_center_integral_tag, comm)
            communication_buffer = Dict{CartesianIndex, Array{Float64, 1}}()
        end
        unlock(communication_buffer_lock)        
    else
        copy_integral_result!(three_center_integrals, integral_buffer, bf_1_pos, bf_2_pos, bf_3_pos, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis)
        axial_normalization_factor(three_center_integrals, shell_1, shell_2, shell_3, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis, bf_1_pos, bf_2_pos, bf_3_pos)
    end
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


@inline function calculate_three_center_integrals_sequential!(three_center_integrals, integral_buffer, cartesian_indices, engine, basis_sets, communication_buffer, communication_buffer_lock)
    for cartesian_index in cartesian_indices
        calculate_three_center_integrals_kernel!(three_center_integrals, engine, cartesian_index, basis_sets, integral_buffer, communication_buffer, communication_buffer_lock)
    end
end

@inline function calculate_three_center_integrals_static(three_center_integrals, cartesian_indices, jeri_engine_thread, basis_sets, thead_integral_buffer, communication_buffer, communication_buffer_lock)
    comm = MPI.COMM_WORLD
    n_ranks = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    

    this_rank_processes_messages = rank == 0 && n_ranks > 1 
    println("static load: rank: $rank, n_ranks: $n_ranks, this_rank_processes_messages: $this_rank_processes_messages")

    rank_shell_indicies = get_df_static_shell_indices(basis_sets,  n_ranks, rank )

    outer_loop_max = length(rank_shell_indicies)
    if  this_rank_processes_messages
        outer_loop_max += 1
    end

    Threads.@sync for thread in 1:outer_loop_max
        Threads.@spawn begin
            if thread == 1 && this_rank_processes_messages
                println("recieving messages for three center intergrals")
                recieve_messages(three_center_integrals, basis_sets)
            else
                threadid = Threads.threadid()
                aux_indicies_index = this_rank_processes_messages ?
                    thread - 1 : # if there is a thread processing messages
                    thread # if there is no thread processing messages on this rank
                aux_index = rank_shell_indicies[aux_indicies_index]
            
                for μ in 1:length(basis_sets.primary)
                    for ν in 1:length(basis_sets.primary)
                        cartesian_index = CartesianIndex(aux_index, μ, ν)
                        engine = jeri_engine_thread[threadid]
                        integral_buffer = thead_integral_buffer[threadid]
                        calculate_three_center_integrals_kernel!(three_center_integrals, engine, cartesian_index, basis_sets, integral_buffer, communication_buffer, communication_buffer_lock)
                    end
                end         
            end
                    
        end
    end
end


@inline function calculate_three_center_integrals_dynamic!(three_center_integrals, cartesian_indices, 
    jeri_engine_thread, basis_sets, thead_integral_buffer, communication_buffer, communication_buffer_lock)
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
            threadid = Threads.threadid()
            if rank == 0 && threadid == 1 && n_ranks > 1
                setup_integral_coordinator(task_top_index, batch_size, n_ranks, n_threads, mutex_mpi_worker)
            else
                run_three_center_integrals_worker(three_center_integrals,
                cartesian_indices,
                batch_size,
                jeri_engine_thread,
                thead_integral_buffer, basis_sets, task_top_index, mutex_mpi_worker, n_indicies, thread, communication_buffer, communication_buffer_lock)
            end
        end
    end
    cleanup_messages()
end


function run_three_center_integrals_worker(three_center_integrals,
    cartesian_indices,
    batch_size,
    jeri_engine_thread,
    thead_integral_buffer, basis_sets, top_index, mutex_mpi_worker, n_indicies, thread, communication_buffer, communication_buffer_lock)

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
        thead_integral_buffer[thread], basis_sets, communication_buffer, communication_buffer_lock)
        ijk_index = get_next_task(mutex_mpi_worker, top_index, batch_size)
    end
end


@inline function do_three_center_integral_batch!(three_center_integrals,
    top_index,
    batch_size,
    cartesian_indices,
    engine,
    integral_buffer, basis_sets, communication_buffer, communication_buffer_lock)
    for ij in top_index:-1:(max(1, top_index - batch_size))
        shell_index = cartesian_indices[ij]
        calculate_three_center_integrals_kernel!(three_center_integrals, engine, shell_index, basis_sets, integral_buffer, communication_buffer, communication_buffer_lock)
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