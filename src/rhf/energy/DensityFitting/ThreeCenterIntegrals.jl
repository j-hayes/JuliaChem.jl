using Base.Threads
using LinearAlgebra
using TensorOperations
using JuliaChem.Shared.Constants.SCF_Keywords
using JuliaChem.Shared

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
    batch_size = ceil(Int, number_of_indices / n_threads)

    max_primary_nbas = max_number_of_basis_functions(basis_sets.primary)
    max_aux_nbas = max_number_of_basis_functions(basis_sets.auxillary)
    thead_integral_buffer = [zeros(Float64, max_primary_nbas^2 * max_aux_nbas) for thread in 1:n_threads]
    if scf_options.load == "sequential"
        calculate_three_center_integrals_sequential!(three_center_integrals, thead_integral_buffer[1], cartesian_indices, jeri_engine_thread[1], basis_sets)
    elseif scf_options.load == "static" || MPI.Comm_size(comm) == 1
        calculate_three_center_integrals_static(three_center_integrals, cartesian_indices, jeri_engine_thread, basis_sets, thead_integral_buffer)
    elseif scf_options.load == "dynamic"
        calculate_three_center_integrals_dynamic!(three_center_integrals, cartesian_indices, jeri_engine_thread, basis_sets, thead_integral_buffer)
    else
        error("integral threading load type: $(scf_options.load) not supported")
    end
    MPI.Barrier(comm)
    three_center_integrals = MPI.Allreduce(three_center_integrals, MPI.SUM, MPI.COMM_WORLD)
    MPI.Barrier(comm)
    return three_center_integrals
end


@inline function calculate_three_center_integrals_kernel!(three_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)
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

    JERI.compute_eri_block_df(engine, integral_buffer, s1, s2, s3, number_of_integrals, 0)

    copy_integral_result!(three_center_integrals, integral_buffer, bf_1_pos, bf_2_pos, bf_3_pos, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis)
    axial_normalization_factor(three_center_integrals, shell_1, shell_2, shell_3, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis, bf_1_pos, bf_2_pos, bf_3_pos)
end

@inline function calculate_three_center_integrals_sequential!(three_center_integrals, integral_buffer, cartesian_indices, engine, basis_sets)
    for cartesian_index in cartesian_indices
        calculate_three_center_integrals_kernel!(three_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)
    end
end

@inline function calculate_three_center_integrals_static(three_center_integrals, cartesian_indices, jeri_engine_thread, basis_sets, thead_integral_buffer)
    comm = MPI.COMM_WORLD
    number_of_indices = length(cartesian_indices)
    comm_size = MPI.Comm_size(comm)
    batch_size = ceil(Int, number_of_indices / (Threads.nthreads()*comm_size))

    stride =  comm_size*batch_size
    start_index = MPI.Comm_rank(comm)*batch_size + 1

    Threads.@sync for batch_index in start_index:stride:number_of_indices
        Threads.@spawn begin
            thread_id = Threads.threadid()
            
            for view_index in batch_index:min(number_of_indices, batch_index + batch_size - 1)
                cartesian_index = cartesian_indices[view_index]
                engine = jeri_engine_thread[thread_id]
                integral_buffer = thead_integral_buffer[thread_id]
                calculate_three_center_integrals_kernel!(three_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)
            end
        end
    end
end


@inline function calculate_three_center_integrals_dynamic!(three_center_integrals, cartesian_indices, jeri_engine_thread, basis_sets, thead_integral_buffer)
    comm = MPI.COMM_WORLD
    n_threads = Threads.nthreads()
    n_pairs = length(cartesian_indices)
    batch_size = size(cartesian_indices, 1)
    rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)
    task_top_index = n_pairs
    if rank == 0
        setup_integral_coordinator(task_top_index, batch_size, n_ranks, n_threads)
    else
    run_three_center_integrals_worker(three_center_integrals,
        cartesian_indices,
        batch_size,
        jeri_engine_thread,
        thead_integral_buffer, basis_sets)
    end
end


@inline function run_three_center_integrals_worker(three_center_integrals,
    cartesian_indices,
    batch_size,
    jeri_engine_thread,
    thead_integral_buffer, basis_sets)

    comm = MPI.COMM_WORLD
    mutex_mpi_worker = Base.Threads.ReentrantLock()
    #== execute kernel ==#
    @sync for thread in 1:Threads.nthreads()
        Threads.@spawn begin
            #== initial set up ==#
            recv_mesg = [0]
            send_mesg = [0, MPI.Comm_rank(comm), thread]
            #== complete first task ==#
            lock(mutex_mpi_worker)
            status = MPI.Probe(0, thread, comm)
            rreq = MPI.Recv!(recv_mesg, status.source, status.tag, comm)
            ij_index = recv_mesg[1]
            unlock(mutex_mpi_worker)

            while ij_index >= 1
                do_three_center_integral_batch!(three_center_integrals,
                ij_index,
                batch_size,
                cartesian_indices,
                jeri_engine_thread[thread],
                thead_integral_buffer[thread], basis_sets)
                ij_index = get_next_batch(mutex_mpi_worker, send_mesg, recv_mesg, comm, thread)
            end
        end
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