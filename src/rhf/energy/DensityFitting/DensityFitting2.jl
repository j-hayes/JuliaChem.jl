#== Density Fitted Restricted Hartree-Fock, Fock build step ==#
#== 
    scf_data::SCFData -> struct with buffers for the DF-RHF SCF calculation e.g. B,V,W,coulomb exchange, etc 
    jeri_engine_thread_df::Vector{T}:: Electron integral engine for each thread 
    jeri_engine_thread ::Vector{T2}:: Electron integral engine for each thread
    basis_sets::CalculationBasisSets -> struct with basis sets for the SCF calculation both primary and auxiliary 
    coefficients -> molecular orbital coefficients
    iteration -> current iteration of the SCF calculation
    scf_options::SCFOptions -> struct with options for the SCF calculation e.g. convergence criteria, screening options, etc
    H::Array{Float64} -> core hamiltonian matrix
    jc_timing::JCTiming -> struct with timing information for the SCF calculation
==#

function df_rhf_fock_build_2!(scf_data::SCFData, jeri_engine_thread_df::Vector{T}, jeri_engine_thread ::Vector{T2},
    basis_sets::CalculationBasisSets,
    coefficients, iteration, scf_options::SCFOptions, H::Array{Float64},
    jc_timing::JCTiming, switch_precision::Bool=false) where {T<:DFRHFTEIEngine, T2<:RHFTEIEngine }

    #todo this should be stored in scf_options instead of directly using Threads.nthreads()
    BLAS.set_num_threads(Threads.nthreads())


    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)
    if iteration == 1 || switch_precision   
        # if rank == 0 && switch_precision
        #     println("switching to precision: ", scf_options.contraction_float_type)
        # end
        aux_basis_function_count = basis_sets.auxillary.norb
        basis_function_count = basis_sets.primary.norb
  
        scf_data.μ = basis_function_count
        scf_data.A = aux_basis_function_count
        scf_data.occ = Int64(basis_sets.primary.nels)÷2

                #ranges for each MPI rank
        shell_aux_indicies, aux_indicies, basis_index_map = static_load_rank_indicies(rank, n_ranks, basis_sets)
        this_rank_Q_range_length = length(aux_indicies)
        scf_options.num_Q_ranges = calculate_on_rank_ranges!(scf_data, scf_options, aux_indicies)
        

        two_center_integrals = calculate_two_center_integrals(jeri_engine_thread_df, basis_sets, scf_options)
        if do_dfrhf_screening(scf_options)
            setup_dfrhf_screening!(scf_data, scf_options, jeri_engine_thread, 
                two_center_integrals, basis_sets, jc_timing)
        else
            setup_unscreened_screening_matricies(basis_sets, scf_data) #allows non-screened 3eri calculation to use same code as screened 3eri
        end
        
        if scf_options.df_use_K_sym
            setup_dfrhf_exchange_blocks!(scf_options.contraction_float_type, scf_data, scf_options, jc_timing)
        end
        
        allocate_dfrhf_memory_cpu!(scf_data, scf_options)
        J_PQ_INV = calculate_J_PQ_inv!(two_center_integrals, scf_options)

        # if scf_options.contraction_float_type != Float64
        #     J_PQ_INV = convert(Array{scf_options.contraction_float_type}, J_PQ_INV)
        # end

        calculate_dfrhf_B!(scf_data, scf_options, J_PQ_INV, basis_sets, 
        jeri_engine_thread_df, jc_timing)
    end

    occupied_orbital_coefficients = get_occupied_orbital_coefficients(scf_data, scf_options, coefficients)
   
    calculate_dfrhf_exchange!(scf_data, scf_options, occupied_orbital_coefficients, jc_timing, iteration)
    calculate_dfrhf_coulomb!(scf_data, scf_options, occupied_orbital_coefficients, jc_timing, iteration)

    if rank == 0
        #add the core hamiltonian to the two electron fock matrix
        H_add_time = @elapsed scf_data.two_electron_fock .+= H
        jc_timing.timings[JCTiming_key(JCTC.H_add_time,iteration)] = H_add_time
    end

    if n_ranks > 1
        #reduce the two electron fock matrix to all ranks
        MPI_time = @elapsed MPI.Allreduce!(scf_data.two_electron_fock, MPI.SUM, comm)  
        jc_timing.timings[JCTiming_key(JCTC.fock_MPI_time,iteration)] = MPI_time
    end
    BLAS.set_num_threads(1)

    # display(scf_data.two_electron_fock)
    # print the fock matrix for debugging in a table format in scientific notation with 8 decimal places
    # for i in 1:scf_data.μ
    #     for j in 1:scf_data.μ
    #         print(@sprintf("%.8e ", scf_data.two_electron_fock[i,j]))
    #     end
    #     println()
    # end
    return scf_data.two_electron_fock   
end

function get_occupied_orbital_coefficients(scf_data::SCFData, scf_options::SCFOptions, coefficients::Array{T,2}) where {T<:Union{Float32, Float64}}
    occupied_orbital_coefficients = coefficients[:,1:scf_data.occ]
    occupied_orbital_coefficients = permutedims(occupied_orbital_coefficients, (2, 1))
    if scf_options.contraction_float_type == Float64
        return occupied_orbital_coefficients        
    end
    occupied_orbital_coefficients_mixed = convert(Array{scf_options.contraction_float_type}, occupied_orbital_coefficients)
    return occupied_orbital_coefficients_mixed
end


function calculate_J_PQ_inv!(two_center_integrals::Array{T}, scf_options::SCFOptions) where {T<:Union{Float32, Float64}}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)

    j_ab_inv_time = @elapsed begin 
        if rank == 0 # avoid convergence problems always do this on rank 0
            # if scf_options.contraction_float_type != Float64
            #     two_center_integrals = scf_options.contraction_float_type.(two_center_integrals)
            # end
            LAPACK.potrf!('L', two_center_integrals)
            LAPACK.trtri!('L', 'N', two_center_integrals)
        end
        if scf_options.contraction_float_type != Float64
            two_center_integrals = scf_options.contraction_float_type.(two_center_integrals)
        end
        if n_ranks > 1
            broadcast_two_center_integrals(two_center_integrals)
        end
    end
    return two_center_integrals
end
# documentation: 
# calculate_dfrhf_B!(scf_data::SCFData, scf_options, two_center_integrals::Array{T}, basis_sets::CalculationBasisSets,
#         jeri_engine_thread_df, jc_timing::JCTiming) where {T <:Union{Float32, Float64}}
#          
# This function calculates the B matrix for the Density Fitted Restricted Hartree-Fock method.
# It uses the two-center integrals and the J_PQ_INV matrix to calculate the B
# matrix for each MPI rank. The B matrix is calculated by performing a matrix multiplication
# between the J_PQ_INV matrix and the three-center integrals. 
# B^Q_{pq} = (pq|P)*J^{-1/2}_{PQ}
function calculate_dfrhf_B!(scf_data::SCFData, scf_options::SCFOptions, J_PQ_INV::Array{T}, basis_sets::CalculationBasisSets, 
        jeri_engine_thread_df, jc_timing::JCTiming) where {T<:Union{Float32, Float64}}

    comm = MPI.COMM_WORLD
    this_rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)
    pq = scf_data.screening_data.screened_indices_count
    three_eri_time = 0.0
    B_time = 0.0

    num_batches = length(scf_data.Q_ranges)
    do_three_eri_screening = do_dfrhf_screening(scf_options)
    if n_ranks == 1 && num_batches == 1
        calculate_dfrhf_B_symmetric(scf_data, scf_options, J_PQ_INV, basis_sets, 
            jeri_engine_thread_df, jc_timing)
        return
    end

     #divide the B_Q indicies that will go to each rank 
    load_balance_indicies = [static_load_rank_indicies_3_eri(rank_index, n_ranks, basis_sets) for rank_index in 0:n_ranks-1]
    three_eri_rank_indicies = load_balance_indicies[this_rank+1][2]
    this_rank_B_Q_index_range = load_balance_indicies[this_rank+1][2]
    
    this_rank_Q_length = length(this_rank_B_Q_index_range)
    J_PQ_INV_for_batches = Vector{Array}(undef, num_batches)

    for ii in 1:num_batches
        this_batch_indicies = this_rank_B_Q_index_range[scf_data.Q_ranges[ii]] # convert from 1-based Q range indicies to indicies in the full 1:num_aux_basis_functions
        J_PQ_INV_for_batches[ii] = J_PQ_INV[this_batch_indicies, :] # this allocates memory perhaps needs to be done another way
    end
    # do B[Q,pq] += J_PQ_INV[Q, P] * three_center_integrals[P,pq] where Q is the aux range managed by this_rank and P is the aux range managed by other_rank(s)
    for other_rank in 0:n_ranks-1
        
        three_eri_time += @elapsed three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, 
            basis_sets,
            scf_options,
            scf_data,
            other_rank,
            n_ranks,
            true, false)
        other_rank_Q_index_range = load_balance_indicies[other_rank+1][2] #range of indexes managed by rank: other rank 

        if scf_options.contraction_float_type != Float64
            three_center_integrals = convert(Array{scf_options.contraction_float_type}, three_center_integrals)
        end

        for batch_index in 1:num_batches
            B_time += @elapsed begin 
                # J_PQ_INV_ranks_slice = view(J_PQ_INV_for_batches[batch_index], :, other_rank_Q_index_range)
                J_PQ_INV_ranks_slice = J_PQ_INV_for_batches[batch_index][:, other_rank_Q_index_range]
                BLAS.gemm!('N', 'N', T(1.0), J_PQ_INV_ranks_slice, three_center_integrals, T(1.0), scf_data.B[batch_index])
            end 
        end
    end

    jc_timing.timings[JCTC.B_time] = B_time
    jc_timing.timings[JCTC.three_eri_time] = three_eri_time
end

#this method is used when there is only one MPI rank and one batch of Q indicies
function calculate_dfrhf_B_symmetric(scf_data::SCFData, scf_options::SCFOptions, J_PQ_INV::Array{T}, basis_sets::CalculationBasisSets,
    jeri_engine_thread_df, jc_timing::JCTiming) where {T<:Union{Float32, Float64}}
    this_rank = 0 
    n_ranks = 1 

    use_screening = do_dfrhf_screening(scf_options)
    three_eri_time = @elapsed scf_data.B[1] .= calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options,
    scf_data, this_rank, n_ranks, true, false)
    # if !use_screening
    #     #reshape for matrix multiplication: todo move this to the three center integral calculation
    #     scf_data.B[1] = reshape(scf_data.B[1], (size(scf_data.B[1],1), size(scf_data.B[1],2)^2))
    # end
    one = scf_options.contraction_float_type(1.0)
    B_time = @elapsed BLAS.trmm!('L', 'L', 'N', 'N', one, J_PQ_INV, scf_data.B[1])    
    jc_timing.timings[JCTC.B_time] = B_time
    jc_timing.timings[JCTC.three_eri_time] = three_eri_time
end


# Auxiliary Ranges can emmerge from two sources 
# 1) MPI Ranks are responsible for a subset of the Auxiliary Ranges
# 2) Auxiliary Ranges are divided into smaller ranges to allow for the ranges created for the Mixed Precision implementation 
# These ranges in the tensor contractions for V,W,J,K are treated as equivalent. Reduction into the Rank Fock Matrix and the 
# Reduction to the other MPI ranks are handled above these contraction functions. 
function allocate_dfrhf_memory_cpu!(scf_data::SCFData, scf_options::SCFOptions)
    T = scf_options.contraction_float_type 
  

    # divide MPI rank ranges into ranges for Mixed Precision
    num_ranges = scf_options.num_Q_ranges

    do_screened = do_dfrhf_screening(scf_options)
    pq = scf_data.μ^2
    if do_screened
        pq = scf_data.screening_data.screened_indices_count
        scf_data.density_array = zeros(T, pq)
    end
    # for each range 
    # allocate B 
    scf_data.B = Vector{Array}(undef, num_ranges)
    scf_data.W_batches = Vector{Array}(undef, num_ranges)
    scf_data.V_batches = Vector{Array}(undef, num_ranges)
    scf_data.J = Vector{Array}(undef, num_ranges)
    scf_data.K = Vector{Array}(undef, num_ranges)
    scf_data.two_electron_fock = zeros(Float64, scf_data.μ, scf_data.μ)
    scf_data.density = zeros(T, scf_data.μ, scf_data.μ)
    
    for ii in 1:num_ranges
        scf_data.B[ii] = zeros(T, length(scf_data.Q_ranges[ii]), pq)
        scf_data.W_batches[ii] = zeros(T, length(scf_data.Q_ranges[ii]), scf_data.occ ,scf_data.μ)
        scf_data.V_batches[ii] = zeros(T, length(scf_data.Q_ranges[ii]))
        scf_data.K[ii] = zeros(T, scf_data.μ, scf_data.μ)
        scf_data.J[ii] = zeros(T, pq)  
    end
    # allocate Fock

end

function calculate_on_rank_ranges!(scf_data, scf_options::SCFOptions, aux_indicies)
    this_rank_Q_range_length = length(aux_indicies)
    num_Q_ranges = get_num_Q_ranges(scf_options, this_rank_Q_range_length)
    scf_options.num_Q_ranges = num_Q_ranges
    scf_data.Q_ranges = Array{UnitRange{Int64}}(undef, num_Q_ranges)
    num_Q_per_range = this_rank_Q_range_length ÷ num_Q_ranges
    #the on rank ranges are indexed starting at 1, if it needs to be adjusted to all rank indicies
    #add to the values based on the start of the aux_indicies for that rank 
    for ii in 1:num_Q_ranges
        scf_data.Q_ranges[ii] = (ii-1)*num_Q_per_range+1:ii*num_Q_per_range
        if ii == num_Q_ranges
            scf_data.Q_ranges[ii] = (ii-1)*num_Q_per_range+1:this_rank_Q_range_length
        end
    end
    return num_Q_ranges
end