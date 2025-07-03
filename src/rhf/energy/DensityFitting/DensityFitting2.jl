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
    jc_timing::JCTiming) where {T<:DFRHFTEIEngine, T2<:RHFTEIEngine }

    println("df_rhf_fock_build_2!")
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)

    
    if iteration == 1 

        aux_basis_function_count = basis_sets.auxillary.norb
        basis_function_count = basis_sets.primary.norb
  
        scf_data.μ = basis_function_count
        scf_data.A = aux_basis_function_count
        scf_data.occ = Int64(basis_sets.primary.nels)÷2


        two_center_integrals = calculate_two_center_integrals(jeri_engine_thread_df, basis_sets, scf_options)
        setup_dfrhf_screening!(scf_data, scf_options, jeri_engine_thread, 
            two_center_integrals, basis_sets, jc_timing)
        allocate_dfrhf_memory_cpu!(scf_data, scf_options, basis_sets)
        J_PQ_INV = calculate_J_PQ_inv!(two_center_integrals)
        calculate_dfrhf_B!(scf_data, scf_options, J_PQ_INV, basis_sets, 
        jeri_engine_thread_df, jc_timing)
    end

    occupied_orbital_coefficients = coefficients[:,1:scf_data.occ]
    if do_screening(scf_options)
        occupied_orbital_coefficients = permutedims(occupied_orbital_coefficients, (2, 1))
    end

    calculate_dfrhf_exchange!(scf_data, scf_options, occupied_orbital_coefficients, jc_timing)
    calculate_dfrhf_coulomb!(scf_data, scf_options, occupied_orbital_coefficients, jc_timing)

    scf_data.two_electron_fock .+= H
    return scf_data.two_electron_fock
end


function calculate_J_PQ_inv!(two_center_integrals::Array{T}) where {T<:Union{Float32, Float64}}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)

    j_ab_inv_time = @elapsed begin 
        if rank == 0 # avoid convergence problems always do this on rank 0
            LAPACK.potrf!('L', two_center_integrals)
            LAPACK.trtri!('L', 'N', two_center_integrals)
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
function calculate_dfrhf_B!(scf_data::SCFData, scf_options, J_PQ_INV::Array{T}, basis_sets::CalculationBasisSets, 
        jeri_engine_thread_df, jc_timing::JCTiming) where {T<:Union{Float32, Float64}}

    comm = MPI.COMM_WORLD
    this_rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)

    
    pq = scf_data.screening_data.screened_indices_count

    three_eri_time = 0.0
    B_time = 0.0

    num_batches = length(scf_data.Q_ranges)


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
        for batch_index in 1:num_batches
            scf_data.B[batch_index] .= 0.0 #shouldn't be necessary but just in case
            B_time += @elapsed begin 
                # this slicing could be on the other dimension and then gemm transposed? TODO(JJH)
                J_PQ_INV_ranks_slice = J_PQ_INV_for_batches[batch_index][:, other_rank_Q_index_range] #this allocates memory perhaps needs to be done another way
                BLAS.gemm!('N', 'N', T(1.0), J_PQ_INV_ranks_slice, three_center_integrals, T(1.0), scf_data.B[batch_index])
            end 
        end
    end

    jc_timing.timings[JCTC.B_time] = B_time
    jc_timing.timings[JCTC.three_eri_time] = three_eri_time
end

#this method is used when there is only one MPI rank and one batch of Q indicies
function calculate_dfrhf_B_symmetric(scf_data::SCFData, scf_options, J_PQ_INV::Array{T}, basis_sets::CalculationBasisSets, 
    jeri_engine_thread_df, jc_timing::JCTiming) where {T<:Union{Float32, Float64}}
    this_rank = 0 
    n_ranks = 1 

    three_eri_time = @elapsed scf_data.B[1] = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options,
    scf_data, this_rank, n_ranks, true, false)
    B_time = @elapsed BLAS.trmm!('L', 'L', 'N', 'N', 1.0, J_PQ_INV, scf_data.B[1])    
    jc_timing.timings[JCTC.B_time] = B_time
    jc_timing.timings[JCTC.three_eri_time] = three_eri_time
end


# Auxiliary Ranges can emmerge from two sources 
# 1) MPI Ranks are responsible for a subset of the Auxiliary Ranges
# 2) Auxiliary Ranges are divided into smaller ranges to allow for the ranges created for the Mixed Precision implementation 
# These ranges in the tensor contractions for V,W,J,K are treated as equivalent. Reduction into the Rank Fock Matrix and the 
# Reduction to the other MPI ranks are handled above these contraction functions. 
function allocate_dfrhf_memory_cpu!(scf_data::SCFData, scf_options::SCFOptions, basis_sets::CalculationBasisSets)
    T = scf_options.contraction_float_type 
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    n_ranks = MPI.Comm_size(MPI.COMM_WORLD) 

    #ranges for each MPI rank
    shell_aux_indicies, aux_indicies, basis_index_map = static_load_rank_indicies(rank, n_ranks, basis_sets)
    this_rank_Q_range_length = length(aux_indicies)

    # divide MPI rank ranges into ranges for Mixed Precision
    num_ranges = calculate_on_rank_ranges!(scf_data, scf_options, aux_indicies)

    do_screened = do_screening(scf_options)
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
    scf_data.density = zeros(Float64, scf_data.μ, scf_data.μ)

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