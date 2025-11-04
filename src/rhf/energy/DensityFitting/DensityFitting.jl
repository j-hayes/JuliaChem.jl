using Base.Threads
using LinearAlgebra
using JuliaChem.Shared.Constants.SCF_Keywords
using JuliaChem.Shared
using Serialization
using HDF5
using ThreadPinning



const BlasInt = LinearAlgebra.BlasInt
const libblastrampoline = LinearAlgebra.libblastrampoline

#== Density Fitted Restricted Hartree-Fock, Fock build step ==#
#== 
indices for all tensor contractions 
  duplicated letters: (e.g.) dd dummy variables that will be summed Overlap
  A = Auxillary Basis orbital
  μ,ν = Primary Basis orbital
  i = occupied orbitals
==#

function df_rhf_fock_build!(scf_data, jeri_engine_thread_df::Vector{T}, jeri_engine_thread ::Vector{T2},
  basis_sets::CalculationBasisSets,
  coefficients, iteration, scf_options::SCFOptions, H::Array{Float64},
  jc_timing::JCTiming) where {T<:DFRHFTEIEngine, T2<:RHFTEIEngine }

  comm = MPI.COMM_WORLD
  rank = MPI.Comm_rank(comm)
  n_ranks = MPI.Comm_size(comm)


  if iteration == 1
    aux_basis_function_count = basis_sets.auxillary.norb
    basis_function_count = basis_sets.primary.norb
    occupied_orbital_count = Int64(basis_sets.primary.nels)÷2

    shell_aux_indicies, aux_indicies, basis_index_map = static_load_rank_indicies(rank, n_ranks, basis_sets)
  
    scf_data.μ = basis_function_count
    scf_data.A = aux_basis_function_count
    scf_data.occ = occupied_orbital_count

    scf_data.two_electron_fock = zeros(Float64, (scf_data.μ, scf_data.μ))
    allocate_memory_density_fitting_dense(scf_data, scf_options, aux_indicies)   
  end

  occupied_orbital_coefficients = coefficients[:,1:scf_data.occ]
 
  if scf_options.contraction_mode == "GPU" || scf_options.contraction_mode == "denseGPU"
    run_gpu_fock_build!(scf_data, jeri_engine_thread_df, jeri_engine_thread, basis_sets, occupied_orbital_coefficients, iteration, scf_options, H, jc_timing)
  else # CPU
    if scf_options.contraction_mode == "dense" || scf_options.df_force_dense 
      # df_rhf_fock_build_BLAS!(scf_data, jeri_engine_thread_df,
      # basis_sets, occupied_orbital_coefficients, iteration, scf_options, jc_timing) 
      if scf_options.do_mixed_precision 
        df_rhf_fock_build_BLAS_mixed_precision!(scf_options.contraction_float_type, scf_data, jeri_engine_thread_df,
        basis_sets, occupied_orbital_coefficients, iteration, scf_options, jc_timing) 
      else
        df_rhf_fock_build_BLAS!(scf_data, jeri_engine_thread_df,
        basis_sets, occupied_orbital_coefficients, iteration, scf_options, jc_timing) 
      end
    else   #default contraction mode is now scf_options.contraction_mode == "screened"
      df_rhf_fock_build_screened!(scf_data, jeri_engine_thread_df, jeri_engine_thread,
      basis_sets, occupied_orbital_coefficients, iteration, scf_options, jc_timing) 
    end
    
    if rank == 0
      H_add_time = @elapsed scf_data.two_electron_fock .+= H # add the core hamiltonian to the two electron fock matrix
      jc_timing.timings[JCTiming_key(JCTC.H_add_time,iteration)] = H_add_time
    end
  end

  if MPI.Comm_size(comm) > 1
    MPI_time = @elapsed MPI.Allreduce!(scf_data.two_electron_fock, MPI.SUM, comm)
    jc_timing.timings[JCTiming_key(JCTC.fock_MPI_time,iteration)] = MPI_time
  end  

  calculate_memory_usage(scf_data, iteration, scf_options, jc_timing)

  return scf_data.two_electron_fock
end

function run_gpu_fock_build!(scf_data, jeri_engine_thread_df, jeri_engine_thread, basis_sets, occupied_orbital_coefficients, iteration, scf_options, H, jc_timing)
  df_force_dense = scf_options.df_force_dense || scf_options.contraction_mode == "denseGPU"
  rank = MPI.Comm_rank(MPI.COMM_WORLD)
  n_ranks = MPI.Comm_size(MPI.COMM_WORLD)

  if df_force_dense || (scf_options.df_use_adaptive && scf_data.μ < scf_options.df_adaptive_basis_limit && rank == 0 && n_ranks == 1) # used for small systems on runs with a single rank
    df_rhf_fock_build_dense_GPU!(scf_data, jeri_engine_thread_df, jeri_engine_thread,
      basis_sets, occupied_orbital_coefficients, iteration, scf_options, H, jc_timing)
  else
    df_rhf_fock_build_GPU!(scf_data, jeri_engine_thread_df, jeri_engine_thread,
      basis_sets, occupied_orbital_coefficients, iteration, scf_options, H, jc_timing)
  end
end


function allocate_memory_density_fitting_dense(scf_data, scf_options, indicies)

  if scf_options.contraction_mode != "dense"
    return
  end

  
  AA = length(indicies)
  μμ = scf_data.μ
  ii = scf_data.occ
  scf_data.D = zeros(Float64, (AA,μμ, μμ))
  scf_data.D_tilde = zeros(Float64, (μμ, AA,ii))

  scf_data.density = zeros(Float64, (μμ, μμ))
  scf_data.coulomb_intermediate = zeros(Float64, AA)
end

function df_rhf_fock_build_BLAS!(scf_data, jeri_engine_thread_df::Vector{T}, basis_sets::CalculationBasisSets,
    occupied_orbital_coefficients, iteration, scf_options::SCFOptions, jc_timing::JCTiming) where {T<:DFRHFTEIEngine}
  comm = MPI.COMM_WORLD
  shell_indicies, aux_indicies, indicies  = static_load_rank_indicies(MPI.Comm_rank(comm),MPI.Comm_size(comm),basis_sets) #todo only do this on iteration 1
  
  if iteration == 1
    two_eri_time = @elapsed two_center_integrals = calculate_two_center_integrals(jeri_engine_thread_df, basis_sets, scf_options)
    calculate_B!(scf_data, two_center_integrals, jc_timing, scf_options, jeri_engine_thread_df, basis_sets)
        
    jc_timing.timings[JCTiming_key(JCTC.two_eri_time,iteration)] = two_eri_time
    jc_timing.non_timing_data[JCTC.contraction_algorithm] = "dense cpu"
  end  
  if iteration == 1
    calculate_coulomb!(scf_data, occupied_orbital_coefficients ,  aux_indicies, jc_timing, iteration)
    temp_fock = copy(scf_data.two_electron_fock)
    #save coulomb to hdf5
    hdf5_file = HDF5.h5open("coulomb.h5", "w")
    write(hdf5_file, "coulomb", scf_data.two_electron_fock)
    println("dimensions of occupied_orbital_coefficients: ", size(occupied_orbital_coefficients))
    write(hdf5_file, "occupied_orbital_coefficients", permutedims(occupied_orbital_coefficients, (2,1)))
    write(hdf5_file, "density", scf_data.density)
    println("dimensions of density: ", size(scf_data.density))
    #write coulomb_intermediate
    write(hdf5_file, "coulomb_intermediate", scf_data.coulomb_intermediate)
    println("dimensions of coulomb_intermediate: ", size(scf_data.coulomb_intermediate))
    HDF5.close(hdf5_file)
    scf_data.two_electron_fock .= 0.0

    calculate_exchange!(scf_data, occupied_orbital_coefficients, aux_indicies, jc_timing, iteration)
    println("dimensions of occupied_orbital_coefficients: ", size(occupied_orbital_coefficients))

    hdf5_file = HDF5.h5open("exchange.h5", "w")
    write(hdf5_file, "exchange", scf_data.two_electron_fock)
    write(hdf5_file, "occupied_orbital_coefficients", permutedims(occupied_orbital_coefficients, (2,1)))
    #write exchange intermediate
    permuted_W = collect(reshape(permutedims(scf_data.D_tilde, (3,2,1)), (scf_data.occ*scf_data.A*scf_data.μ)))
    write(hdf5_file, "exchange_intermediate", permuted_W)
    println("dimensions of exchange_intermediate: ", size(scf_data.D_tilde))
    HDF5.close(hdf5_file)




    scf_data.two_electron_fock .+= temp_fock
  
    #write fock to hdf5 
    hdf5_file = HDF5.h5open("fock.h5", "w")
    write(hdf5_file, "fock", scf_data.two_electron_fock)
    HDF5.close(hdf5_file)
  end
  calculate_coulomb!(scf_data, occupied_orbital_coefficients ,  aux_indicies, jc_timing, iteration)
  calculate_exchange!(scf_data, occupied_orbital_coefficients, aux_indicies, jc_timing, iteration)
  
end


function calculate_B!(scf_data, two_center_integrals, jc_timing::JCTiming,
  scf_options::SCFOptions, jeri_engine_thread_df::Vector{T},
  basis_sets::CalculationBasisSets) where {T<:DFRHFTEIEngine}
  μμ = scf_data.μ
  νν = scf_data.μ

  n_ranks = MPI.Comm_size(MPI.COMM_WORLD)
  rank = MPI.Comm_rank(MPI.COMM_WORLD)
  J_AB_invt = zeros(scf_options.contraction_float_type, (2,2))

   #write two center integrals to an HDF5 file called two_center_integrals.h5 for debugging
  hdf5_file = HDF5.h5open("two_center_integrals.h5", "w")
  write(hdf5_file, "two_center_integrals", two_center_integrals)

  form_J_AB_inv_time = @elapsed begin
    if rank == 0 # avoid convergence problems always do this on rank 0
      # if scf_options.contraction_float_type != Float64
      #   two_center_integrals = scf_options.contraction_float_type.(two_center_integrals)
      # end
      LAPACK.potrf!('L', two_center_integrals)
      LAPACK.trtri!('L', 'N', two_center_integrals)
    end

    #write two center integrals to HDF5 for debugging 
    
    write(hdf5_file, "J_PQ_inv", two_center_integrals)

    J_AB_invt = zeros(scf_options.contraction_float_type, size(two_center_integrals))
    J_AB_invt .= two_center_integrals

    if n_ranks > 1
        broadcast_two_center_integrals(J_AB_invt)
    end

  end


  B_time = 0.0
  three_eri_time = 0.0
  three_center_integrals = []
  one = scf_options.contraction_float_type(1.0)
  zero = scf_options.contraction_float_type(0.0)
  
  scf_data.D = zeros(scf_options.contraction_float_type, size(scf_data.D))

  hdf5_file = HDF5.h5open("three_center_integrals.h5", "w")

  if n_ranks == 1  #single rank case
    AA = scf_data.A
    three_eri_time = @elapsed three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options, 
    scf_data, rank,n_ranks, false, false)

    #write three_center_integrals to hdf5
    # three_eri_for_hdf5 = permutedims(three_center_integrals, (2,3,1))
    println("dimensions of three center integrals: ", size(three_center_integrals))
    write(hdf5_file, "three_center_integrals", reshape(three_center_integrals, (AA,μμ*νν)))

    scf_data.D = zeros(scf_options.contraction_float_type, size(three_center_integrals))
    scf_data.D .= three_center_integrals
    B_time = @elapsed BLAS.trmm!('L', 'L', 'N', 'N', one, two_center_integrals, reshape(scf_data.D, (AA, μμ * νν)))

    #write B to hdf5
    # B_for_hdf5 = permutedims(scf_data.D, (2,3,1))
    println("dimensions of B: ", size(scf_data.D))
    write(hdf5_file, "B", reshape(scf_data.D, (AA, μμ*νν)))
  else

    setup_unscreened_screening_matricies(basis_sets, scf_data)

    rank_shell_aux_indicies, 
    rank_aux_indicies, 
    rank_basis_index_map = static_load_rank_indicies(rank,n_ranks,basis_sets) 
    AA = length(rank_aux_indicies)

    scf_data.D = zeros(Float64, (AA, μμ * νν))
    form_J_AB_inv_time += @elapsed this_rank_two_eri = two_center_integrals[rank_aux_indicies,:]

    for other_rank in 0:n_ranks-1
      other_rank_shell_aux_indicies, 
      other_rank_aux_indicies,
      other_rank_basis_index_map = static_load_rank_indicies(other_rank,n_ranks,basis_sets) 

      three_eri_time += @elapsed begin
        three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options,
          scf_data, other_rank, n_ranks, true, false)
      end
      B_time += @elapsed BLAS.gemm!('N', 'N', 1.0, this_rank_two_eri[:,other_rank_aux_indicies], three_center_integrals, 1.0, scf_data.D)
    end

    #print the first row of the three center integrals
  
  end

  jc_timing.timings[JCTC.form_J_AB_inv_time] = form_J_AB_inv_time
  jc_timing.timings[JCTC.B_time] = B_time
  jc_timing.timings[JCTC.three_eri_time] = three_eri_time
end

function calculate_coulomb!(scf_data, occupied_orbital_coefficients, indicies, jc_timing::JCTiming, iteration)
  Q = length(indicies)
  pq = scf_data.μ^2
  B = scf_data.D
  V = scf_data.coulomb_intermediate
  fock = scf_data.two_electron_fock
  density = scf_data.density 

  BLAS_threads = Base.Threads.nthreads()

  blas_threads = BLAS.get_num_threads()
  if scf_data.μ < 200 
      BLAS.set_num_threads(1)
  end
  density_time = @elapsed BLAS.gemm!('N', 'T', 1.0, occupied_orbital_coefficients, occupied_orbital_coefficients, 0.0, density)
  BLAS.set_num_threads(BLAS_threads)

  V_time = @elapsed begin
    BLAS.gemv!('N', 1.0, reshape(B, (Q, pq)), reshape(density, pq), 0.0, V)
  end
  J_time = @elapsed begin
    BLAS.gemv!('T', 2.0, reshape(B, (Q, pq)), V, 0.0, reshape(fock, pq))
  end
  jc_timing.timings[JCTiming_key(JCTC.density_time,iteration)] = density_time
  jc_timing.timings[JCTiming_key(JCTC.V_time,iteration)] = V_time
  jc_timing.timings[JCTiming_key(JCTC.J_time,iteration)] = J_time
end



function df_rhf_fock_build_BLAS_mixed_precision!(FloatT::Type, scf_data, jeri_engine_thread_df::Vector{T}, basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions, jc_timing::JCTiming) where {T<:DFRHFTEIEngine}
comm = MPI.COMM_WORLD
shell_indicies, aux_indicies, indicies  = static_load_rank_indicies(MPI.Comm_rank(comm),MPI.Comm_size(comm),basis_sets) #todo only do this on iteration 1





if iteration == 1
  println("doing mixed precision $(scf_options.contraction_float_type) DF-RHF tensor contractions")
  two_eri_time = @elapsed two_center_integrals = calculate_two_center_integrals(jeri_engine_thread_df, basis_sets, scf_options)
  calculate_B!(scf_data, two_center_integrals, jc_timing, scf_options, jeri_engine_thread_df, basis_sets)
  B = zeros(FloatT, (scf_data.μ, scf_data.μ, scf_data.A))
  B .= permutedims(scf_data.D, (2,3,1))
  scf_data.D = B

      
  jc_timing.timings[JCTiming_key(JCTC.two_eri_time,iteration)] = two_eri_time
  jc_timing.non_timing_data[JCTC.contraction_algorithm] = "dense cpu"
  scf_data.J = zeros(FloatT, (scf_data.μ, scf_data.μ))
  scf_data.K = zeros(FloatT, (scf_data.μ, scf_data.μ))
  scf_data.density = zeros(FloatT, (scf_data.μ, scf_data.μ))

end  


num_Q_ranges = get_num_Q_ranges(scf_options, scf_data.A)


Q = scf_data.A 
Q_ranges = []
scf_data.D_tilde = Vector{Array}(undef, num_Q_ranges)
scf_data.coulomb_intermediate = Vector{Array}(undef, num_Q_ranges)
for q_range_index in 1:num_Q_ranges
    start = (q_range_index-1) * div(Q, num_Q_ranges) + 1
    stop = q_range_index * div(Q, num_Q_ranges)
    if q_range_index == num_Q_ranges
        stop = Q
    end
    push!(Q_ranges, start:stop)
    scf_data.D_tilde[q_range_index] = zeros(FloatT, (scf_data.μ, length(Q_ranges[q_range_index]), scf_data.occ))
    scf_data.coulomb_intermediate[q_range_index] = zeros(FloatT, length(Q_ranges[q_range_index]))
end


occupied_orbital_coefficients_mixed = zeros(FloatT, size(occupied_orbital_coefficients))
occupied_orbital_coefficients_mixed .= occupied_orbital_coefficients
scf_data.two_electron_fock .= 0.0

calculate_exchange_mixed_precision!(FloatT, scf_data, occupied_orbital_coefficients_mixed, iteration, Q_ranges, num_Q_ranges)
calculate_coulomb_mixed_precision!(FloatT, scf_data, occupied_orbital_coefficients_mixed, iteration, Q_ranges, num_Q_ranges)


end


function calculate_coulomb_mixed_precision!(FloatT::Type, scf_data, occupied_orbital_coefficients, iteration, Q_ranges, num_Q_ranges)
  Q = scf_data.A
  pq = scf_data.μ^2
  B = scf_data.D
  V = scf_data.coulomb_intermediate
  fock = scf_data.two_electron_fock
  density = scf_data.density 

  BLAS_threads = Base.Threads.nthreads()
  blas_threads = BLAS.get_num_threads()
  if scf_data.μ < 200 
      BLAS.set_num_threads(1)
  end


  one_mixed = FloatT(1.0)
  two_mixed = FloatT(2.0)
  neg_mixed = FloatT(-1.0)
  zero_mixed = FloatT(0.0)

  if scf_data.μ < 200 
      BLAS.set_num_threads(1)
  end

  BLAS.gemm!('N', 'T', one_mixed, occupied_orbital_coefficients, occupied_orbital_coefficients, zero_mixed, density)
  BLAS.set_num_threads(BLAS_threads)
  for q_range_index in 1:num_Q_ranges
    Q_range = Q_ranges[q_range_index]

    B_reshape = reshape(view(B, :, :, Q_range), (pq, length(Q_range)))

    BLAS.gemv!('T', one_mixed, B_reshape, reshape(density, pq), zero_mixed, V[q_range_index])
    BLAS.gemv!('N', two_mixed, B_reshape, V[q_range_index], zero_mixed, reshape(scf_data.J, pq))
    scf_data.two_electron_fock .+= scf_data.J
  end
end

function calculate_exchange_mixed_precision!(FloatT::Type, scf_data, occupied_orbital_coefficients, iteration, Q_ranges, num_Q_ranges)
  Q = scf_data.A
  p = scf_data.μ
  n_ooc = scf_data.occ
  K = scf_data.K

  # ooc = occupied_orbital_coefficients
  B = scf_data.D
  W = scf_data.D_tilde
  fock = scf_data.two_electron_fock


  one_mixed = FloatT(1.0)
  neg_one_mixed = FloatT(-1.0)
  zero_mixed = FloatT(0.0)

  for q_range_index in 1:num_Q_ranges
    Q_range = Q_ranges[q_range_index]
    W_reshape = reshape(scf_data.D_tilde[q_range_index], (p*length(Q_range), n_ooc))
    B_reshape = reshape(view(B, :, :, Q_range), (p, p*length(Q_range)))
    BLAS.gemm!('T', 'N', one_mixed, B_reshape, occupied_orbital_coefficients, zero_mixed, W_reshape)
    W_reshape_K = reshape(scf_data.D_tilde[q_range_index], (p, n_ooc * length(Q_range)))
    BLAS.gemm!('N', 'T', neg_one_mixed, W_reshape_K, W_reshape_K, zero_mixed, K)

    scf_data.two_electron_fock .+= K
  end

end



function calculate_exchange!(scf_data, occupied_orbital_coefficients, indicies, jc_timing::JCTiming, iteration)
  Q = length(indicies)
  p = scf_data.μ
  n_ooc = scf_data.occ

  ooc = occupied_orbital_coefficients
  B = scf_data.D
  W = scf_data.D_tilde
  fock = scf_data.two_electron_fock


  W_time = @elapsed begin
    BLAS.gemm!('T', 'T', 1.0, ooc, reshape(B, (Q * p, p)), 0.0, reshape(W, (n_ooc, Q * p)))
  end
  K_time = @elapsed begin
    BLAS.gemm!('T', 'N', -1.0, reshape(W, (n_ooc * Q, p)), reshape(W, (n_ooc * Q, p)), 1.0, fock)
  end
  jc_timing.timings[JCTiming_key(JCTC.W_time,iteration)] = W_time
  jc_timing.timings[JCTiming_key(JCTC.K_time,iteration)] = K_time

end

function calculate_memory_usage(scf_data, iteration, scf_options, jc_timing)
  jc_timing.non_timing_data[JCTiming_key(JCTC.scf_data_size_MB, iteration)] = string(Base.summarysize(scf_data) / 1024^2)
end