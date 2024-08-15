using Base.Threads
using LinearAlgebra
using TensorOperations
using JuliaChem.Shared.Constants.SCF_Keywords
using JuliaChem.Shared
using MPI 

three_center_integral_tag = 3000

#no screening
function calculate_three_center_integrals(jeri_engine_thread, basis_sets::CalculationBasisSets, scf_options::SCFOptions)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    n_ranks = MPI.Comm_size(MPI.COMM_WORLD)
    return calculate_three_center_integrals(jeri_engine_thread, basis_sets, scf_options, SCFData(), rank, n_ranks) ##todo this should all be one interface after refactor
end

function calculate_three_center_integrals(jeri_engine_thread, basis_sets::CalculationBasisSets, scf_options::SCFOptions, 
    scf_data::SCFData)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    n_ranks = MPI.Comm_size(MPI.COMM_WORLD)
    return calculate_three_center_integrals(jeri_engine_thread, basis_sets, scf_options, scf_data, rank, n_ranks)
end

function calculate_three_center_integrals(jeri_engine_thread, basis_sets::CalculationBasisSets, scf_options::SCFOptions, 
    scf_data::SCFData, rank::Int, n_ranks::Int)
    aux_basis_function_count = basis_sets.auxillary.norb
    basis_function_count = basis_sets.primary.norb
    auxilliary_basis_shell_count = length(basis_sets.auxillary)
    basis_shell_count = length(basis_sets.primary)

    n_threads = Threads.nthreads()
    three_center_integrals = []
    max_primary_nbas = max_number_of_basis_functions(basis_sets.primary)
    max_aux_nbas = max_number_of_basis_functions(basis_sets.auxillary)
    thead_integral_buffer = [zeros(Float64, max_primary_nbas^2 * max_aux_nbas) for thread in 1:n_threads]
    if scf_options.load == "screened"
        three_center_integrals = calculate_three_center_integrals_screened!(rank, n_ranks, 
            jeri_engine_thread, basis_sets, thead_integral_buffer, 
            scf_data.screening_data.screened_indices_count,
            scf_data.screening_data.shell_screen_matrix, 
            scf_data.screening_data.sparse_pq_index_map)
    else
        three_center_integrals = zeros(Float64, (basis_function_count, basis_function_count, aux_basis_function_count))
        if scf_options.load == "sequential"
            cartesian_indices = CartesianIndices((auxilliary_basis_shell_count, basis_shell_count, basis_shell_count))
            calculate_three_center_integrals_sequential!(three_center_integrals, thead_integral_buffer[1], shell_screen_matrix, sparse_pq_index_map, cartesian_indices, jeri_engine_thread[1], basis_sets)
        elseif scf_options.load == "static"
            calculate_three_center_integrals_static(three_center_integrals, jeri_engine_thread, basis_sets, thead_integral_buffer)
        elseif scf_options.load == "dynamic"
            calculate_three_center_integrals_dynamic!(three_center_integrals, jeri_engine_thread, basis_sets, thead_integral_buffer)
        else
            error("integral threading load type: $(scf_options.load) not supported")
        end
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

    #todo make the cartesian index match how the array is stored (μ ,ν, A)

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

#calculate only the integrals that pass Schwarz screening
#sparse_pq_index_map is a map of the non screened pq pairs to where in the screned matrix they should go in the 1D pq index, could be dense or triangular 
@inline function calculate_three_center_integrals_kernel_screened!(three_center_integrals, engine, cartesian_index, basis_sets, integral_buffer, sparse_pq_index_map, rank_basis_index_map)
    s1, s2, s3,
    shell_1, shell_2, shell_3,
    shell_1_nbasis, shell_2_nbasis, shell_3_nbasis, 
    bf_1_pos, bf_2_pos, bf_3_pos, 
    number_of_integrals = get_indexes_eri_block(cartesian_index, basis_sets)
    
    JERI.compute_eri_block_df(engine, integral_buffer, s1, s2, s3, number_of_integrals, 0)
    copy_integral_result_screened!(three_center_integrals, integral_buffer, bf_1_pos, bf_2_pos, bf_3_pos, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis, sparse_pq_index_map, rank_basis_index_map)
    axial_normalization_factor_screened!(three_center_integrals, shell_1, shell_2, shell_3, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis, bf_1_pos, bf_2_pos, bf_3_pos, sparse_pq_index_map, rank_basis_index_map)

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


function calculate_three_center_integrals_sequential!(three_center_integrals, integral_buffer, shell_screen_matrix, basis_screen_matrix, cartesian_indices, engine, basis_sets)
    for cartesian_index in cartesian_indices
        calculate_three_center_integrals_kernel!(three_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)
    end
end


function calculate_three_center_integrals_static(three_center_integrals, jeri_engine_thread, basis_sets, thead_integral_buffer)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)
    nthreads = Threads.nthreads()
    basis_length = length(basis_sets.primary)
    
    load_balance_indicies = [static_load_rank_indicies(rank_index, n_ranks, basis_sets) for rank_index in 0:n_ranks-1]
    rank_shell_indicies = load_balance_indicies[rank+1][1] 
    rank_basis_indicies = load_balance_indicies[rank+1][2] 

    rank_number_of_shells = length(rank_shell_indicies)
    n_indicies_per_thread = rank_number_of_shells÷nthreads
    Threads.@sync for thread in 1:nthreads
        Threads.@spawn begin     
            thread_index_offset = static_load_thread_index_offset(thread, n_indicies_per_thread)
            n_shells_to_process = static_load_thread_shell_to_process_count(thread, nthreads, rank_number_of_shells, n_indicies_per_thread)

            for shell_index in 1:n_shells_to_process
                aux_index = rank_shell_indicies[shell_index + thread_index_offset]
                for μ in 1:basis_length
                    for ν in 1:μ 
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
        # MPI.Allreduce!(three_center_integrals, MPI.SUM, comm)
        gather_and_reduce_three_center_integrals(three_center_integrals, load_balance_indicies, rank_basis_indicies, comm)
    end
end


function gather_and_reduce_three_center_integrals(three_center_integrals, load_balance_indicies, rank_basis_indicies, comm)
    rank = MPI.Comm_rank(comm)
    number_of_primary_basis_functions = size(three_center_integrals, 1)
    aux_basis_indicies_per_rank = [length(x[2]) for x in load_balance_indicies] # number of basis functions calculated on each 
    rank_indicies = [x[2] ::Array{Int} for x in load_balance_indicies] #basis function indicies calculated on each 
    indicies_per_rank = aux_basis_indicies_per_rank.*(number_of_primary_basis_functions^2)
    three_center_integral_buff = MPI.VBuffer(three_center_integrals, indicies_per_rank) # buffer set up with the correct size for each rank
    MPI.Allgatherv!(three_center_integrals[:,:,rank_basis_indicies], three_center_integral_buff, comm) # gather the data from each rank into the buffer
    reorder_mpi_gathered_matrix(three_center_integrals, 
        rank_indicies, set_data_3D!, set_temp_3D!, zeros(Float64,number_of_primary_basis_functions , number_of_primary_basis_functions))
end



function calculate_three_center_integrals_dynamic!(three_center_integrals, 
    jeri_engine_thread, basis_sets, thead_integral_buffer)
    comm = MPI.COMM_WORLD
    n_threads = Threads.nthreads()
    rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)
    n_aux_shells = length(basis_sets.auxillary)
    n_primary_shells = length(basis_sets.primary)
    number_of_primary_basis_functions = size(three_center_integrals, 1)
    
    # all threads get pre determined first index to process this is the next lowest index for processing
    top_index, aux_indicies_processed = setup_dynamic_load_indicies(n_aux_shells, n_ranks)
    aux_shell_index_to_process = aux_indicies_processed[rank+1][1] # each rank starts with the (aux_shell_index_to_process = n_aux_shells - rank) a predetermined index to process first to start load balancing evenly without an extra mpi message.
    n_worker_threads = get_number_of_dynamic_worker_threads(rank, n_ranks)
    
    if n_worker_threads > n_primary_shells
        n_worker_threads = n_primary_shells
    end

    @sync begin 
        mutex_mpi_worker = Base.Threads.ReentrantLock() # lock for the use of the messaging and the top_index variable
        if rank == 0 && n_ranks > 1
            Threads.@spawn setup_integral_coordinator_aux(three_center_integral_tag,
                mutex_mpi_worker, top_index, aux_indicies_processed)
        end
        while true
            @sync begin
                for thread in 1:n_worker_threads
                    Threads.@spawn begin
                        number_of_indicies_to_process = n_primary_shells ÷ n_threads
                        start_index = (thread - 1) * number_of_indicies_to_process + 1
                        end_index = start_index + number_of_indicies_to_process - 1
                        if thread == n_worker_threads
                            end_index = n_primary_shells
                        end
                        integral_buffer = thead_integral_buffer[thread]
                        engine = jeri_engine_thread[thread]
                        for j_index in start_index:end_index # proecess the threads portion of the second index
                            for i_index in 1:n_primary_shells # process all of the first index for (j_index,aux_shell_index_to_process) pair
                                shell_index = CartesianIndex(aux_shell_index_to_process, i_index, j_index)
                                calculate_three_center_integrals_kernel!(three_center_integrals, engine, shell_index, basis_sets, integral_buffer)
                            end
                        end
                    end
                end
            end
            lock(mutex_mpi_worker) do 
                aux_shell_index_to_process = get_next_task_aux!(top_index, three_center_integral_tag, aux_indicies_processed, rank)
            end
            if aux_shell_index_to_process < 1
                break
            end
        end
    end
    if n_ranks > 1
        aux_indicies_processed = broadcast_processed_index_list(aux_indicies_processed, n_ranks, n_aux_shells)
        rank_basis_indices, indicies_per_rank = get_allranks_basis_indicies_for_shell_indicies!(aux_indicies_processed, n_ranks,basis_sets, number_of_primary_basis_functions^2)       
        three_center_integral_buff = MPI.VBuffer(three_center_integrals, indicies_per_rank) # buffer set up with the correct size for each rank
        MPI.Allgatherv!(three_center_integrals[ :,:,rank_basis_indices[rank+1]], three_center_integral_buff, comm) # gather the data from each rank into the buffer
        reorder_mpi_gathered_matrix(three_center_integrals, rank_basis_indices, set_data_3D!, set_temp_3D!, zeros(Float64, number_of_primary_basis_functions , number_of_primary_basis_functions))
    end
end




function calculate_three_center_integrals_screened!(rank, n_ranks, 
    jeri_engine_thread, basis_sets, thead_integral_buffer, screened_pq_index_count, shell_screen_matrix, sparse_pq_index_map)
    

   
    nthreads = Threads.nthreads()
    basis_length = length(basis_sets.primary)
    
    rank_shell_indicies, 
    rank_basis_indicies, 
    rank_basis_index_map = static_load_rank_indicies_3_eri(rank, n_ranks, basis_sets) 
    


    rank_number_of_shells = length(rank_shell_indicies)
    n_indicies_per_thread = rank_number_of_shells÷nthreads

    rank_P = length(rank_basis_indicies) #the total number of indicies given to this rank
    three_center_integrals = zeros(Float64, (rank_P, screened_pq_index_count))

    Threads.@sync for thread in 1:nthreads
        Threads.@spawn begin     
            thread_index_offset = static_load_thread_index_offset(thread, n_indicies_per_thread)
            n_shells_to_process = static_load_thread_shell_to_process_count(thread, nthreads, rank_number_of_shells, n_indicies_per_thread)

            for shell_index in 1:n_shells_to_process
                
                aux_index = rank_shell_indicies[shell_index + thread_index_offset]
                for μ in 1:basis_length
                    for ν in 1:μ 
                        if shell_screen_matrix[μ, ν] == false
                            continue 
                        end
                        cartesian_index = CartesianIndex(aux_index, μ, ν)
                        engine = jeri_engine_thread[thread]
                        integral_buffer = thead_integral_buffer[thread]
                        calculate_three_center_integrals_kernel_screened!(three_center_integrals, engine, cartesian_index, basis_sets, integral_buffer, sparse_pq_index_map, rank_basis_index_map)
                    end
                end    
            end
        end
    end
    return three_center_integrals
end


@inline function copy_integral_result!(three_center_integrals, values, bf_1_pos, bf_2_pos, bf_3_pos, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis)
    values_index = 1
    for i in bf_1_pos:bf_1_pos+shell_1_nbasis-1
        for j in bf_2_pos:bf_2_pos+shell_2_nbasis-1
            for k in bf_3_pos:bf_3_pos+shell_3_nbasis-1
                three_center_integrals[j, k, i] = values[values_index]

                if j >= k
                    three_center_integrals[j, k, i] = values[values_index]
                else
                    three_center_integrals[j, k, i] = 0.0
                end
                values_index += 1
            end
        end
    end
end

@inline function copy_integral_result_screened!(three_center_integrals, values, bf_1_pos, bf_2_pos, bf_3_pos, 
    shell_1_nbasis, shell_2_nbasis, shell_3_nbasis, 
    screen_index_matrix, rank_basis_index_map)
    values_index = 1

    for i in bf_1_pos:bf_1_pos+shell_1_nbasis-1 #auxiliary_basis index
        rank_i = rank_basis_index_map[i] #auxiliary_basis index for the rank 
        for j in bf_2_pos:bf_2_pos+shell_2_nbasis-1
            for k in bf_3_pos:bf_3_pos+shell_3_nbasis-1
                screened_index = screen_index_matrix[j,k]
                if screened_index == 0
                    values_index += 1
                    continue
                end
                three_center_integrals[rank_i,screened_index] = values[values_index]
                values_index += 1
            end
        end
    end
end

function print_three_center_integrals(three_center_integrals)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    println("three center integrals\n")
    i = 0
    io = open(joinpath(@__DIR__,  "./three_center_integrals_out-rank-$rank.txt"), "w+")
    for index in CartesianIndices(three_center_integrals)
        write(io,"3-ERI[$(index[1]),$(index[2]),$(index[3])] = $(three_center_integrals[index])\n")
        i += 1
        # if i > 10000 
        #     break
        # end
    end
    close(io)
end