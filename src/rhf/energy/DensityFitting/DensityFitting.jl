using Base.Threads
using LinearAlgebra
using TensorOperations
using JuliaChem.Shared.Constants.SCF_Keywords
using JuliaChem.Shared
#== Density Fitted Restricted Hartree-Fock, Fock build step ==#
#== 
Indecies for all tensor contractions 
  duplicated letters: (e.g.) dd dummy variables that will be summed Overlap
  A = Auxillary Basis orbital
  μ,ν = Primary Basis orbital
  i = occupied orbitals
==#

@inline function df_rhf_fock_build!(scf_data, jeri_engine_thread::Vector{T}, 
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}
  comm = MPI.COMM_WORLD
  if iteration == 1
    two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread, basis_sets, scf_options)
    three_center_integrals = calculate_three_center_integrals(jeri_engine_thread, basis_sets, scf_options)
    calculate_D!(scf_data, two_center_integrals, three_center_integrals, scf_options)
  end
    calculate_D_tilde!(scf_data, occupied_orbital_coefficients, scf_options)
    calculate_coulomb!(scf_data, occupied_orbital_coefficients, scf_options) 
    calculate_exchange!(scf_data,scf_options)
  
  MPI.Barrier(comm)
end

#original tensor operation  @tensor two_electron_fock[μ, ν] -= xiK[μ, νν, AA] * xiK[ν, νν, AA]
function calculate_exchange!(scf_data, scf_options::SCFOptions)
  if scf_options.contraction_mode == ContractionMode.tensor_operations
    @tensor scf_data.two_electron_fock[μ, ν] -= scf_data.D_tilde[μ, νν, AA] * scf_data.D_tilde[ν, νν, AA]
    return 
  end

  # todo get these from constants to make this more readable 
  μμ = scf_data.μ # number of basis functions 
  oo = scf_data.occ # number of occupied orbitals
  AA = scf_data.A # number of aux basis functions

  # transpose transpose gemm transpose tensor contraction 
  μ_vector =  reshape(scf_data.D_tilde, (μμ, oo*AA))
  ν_vector =  reshape(scf_data.D_tilde, (μμ, oo*AA))
  BLAS.gemm!('N', 'T', -1.0, μ_vector, ν_vector, 1.0,scf_data.two_electron_fock)

end

@inline function calculate_coulomb!(scf_data, occupied_orbital_coefficients,
   scf_options::SCFOptions) 
     
  if scf_options.contraction_mode == ContractionMode.tensor_operations 
    @tensoropt (μμ => 10, ii => 10) scf_data.two_electron_fock[μ, ν] = 2.0 * scf_data.D[μ, ν, A] * 
    scf_data.D_tilde[μμ,A,ii] * occupied_orbital_coefficients[μμ, ii]
    return 
  end
  scf_data.D_tilde_permuted = permutedims(scf_data.D_tilde, (2,1,3))

  μμ = scf_data.μ
  νν = scf_data.μ
  ii = scf_data.occ
  AA = scf_data.A

  scf_data.D_tilde_permuted = reshape(scf_data.D_tilde_permuted, (AA,μμ*ii))

  occupied_orbital_coefficients_vector = reshape(occupied_orbital_coefficients, (μμ*ii,1))
  BLAS.gemm!('N', 'N', 1.0, scf_data.D_tilde_permuted, occupied_orbital_coefficients_vector ,0.0, scf_data.coulomb_intermediate)

  scf_data.D = reshape(scf_data.D, (μμ*νν, AA))
  scf_data.two_electron_fock = reshape(scf_data.two_electron_fock, (μμ*νν, 1))
  BLAS.gemm!('N', 'N', 2.0, scf_data.D, scf_data.coulomb_intermediate, 0.0 ,scf_data.two_electron_fock)
  scf_data.two_electron_fock = reshape(scf_data.two_electron_fock, (μμ, νν)) 

end 

# calculate xiK 
@inline function calculate_D_tilde!(scf_data, occupied_orbital_coefficients, scf_options::SCFOptions)
  if scf_options.contraction_mode == ContractionMode.tensor_operations
    @tensor scf_data.D_tilde[ν, A, i] := scf_data.D[μμ, ν, A] * occupied_orbital_coefficients[μμ, i]
    return
  end 
  μμ = scf_data.μ 
  AA = scf_data.A
  ii = scf_data.occ
  
  scf_data.D_tilde = reshape(scf_data.D_tilde, (μμ*AA,ii))
  BLAS.gemm!('N', 'N' , 1.0, scf_data.D_permuted, occupied_orbital_coefficients, 0.0, scf_data.D_tilde)
  scf_data.D_tilde = reshape(scf_data.D_tilde, (μμ,AA,ii))
end


@inline function calculate_D!(scf_data, two_center_integrals, three_center_integrals, scf_options::SCFOptions)
  comm = MPI.COMM_WORLD
  # this needs to be mpi parallelized
  flush(stdout)
  J_AB_invt = convert(Array, transpose(cholesky(Hermitian(two_center_integrals, :L)).L \I))


  if scf_options.contraction_mode == ContractionMode.tensor_operations
    @tensor scf_data.D[μ, ν, A] = three_center_integrals[μ, ν, BB]*J_AB_invt[BB, A]
    return
  end
  
  ## todo get this from a parameter to make this more readable 
  μμ = scf_data.μ
  νν = scf_data.μ
  AA = scf_data.A
  # use transpose transpose gemm transpose to perform tensor contraction 
  # Linv_T is already a 2D matrix so no need to reshape, and is in correct order
  scf_data.D = reshape(scf_data.D, (μμ*νν,AA))
  three_center_integrals =  reshape(three_center_integrals, (μμ*νν,AA))
  BLAS.gemm!('N', 'N', 1.0, three_center_integrals, J_AB_invt, 0.0, scf_data.D)
  scf_data.D = reshape(scf_data.D, (μμ, νν, AA))
  permutedims!(scf_data.D_permuted, scf_data.D, (1,3,2))
  scf_data.D_permuted = reshape(scf_data.D_permuted, (νν*AA, μμ))
  # MPI.Barrier(comm)
end # end function calculate_D

@inline function calculate_three_center_integrals(jeri_engine_thread, basis_sets::CalculationBasisSets, scf_options::SCFOptions)
  comm = MPI.COMM_WORLD

  aux_basis_function_count = basis_sets.auxillary.norb
  basis_function_count = basis_sets.primary.norb
  three_center_integrals = zeros(Float64, (basis_function_count, basis_function_count, aux_basis_function_count))
  auxilliary_basis_shell_count = length(basis_sets.auxillary)
  basis_shell_count = length(basis_sets.primary)

  cartesian_indecies = CartesianIndices((auxilliary_basis_shell_count, basis_shell_count, basis_shell_count))
  number_of_indecies = length(cartesian_indecies)
  n_threads = Threads.nthreads()
  batch_size = ceil(Int, number_of_indecies / n_threads)

  max_primary_nbas = max_number_of_basis_functions(basis_sets.primary)
  max_aux_nbas = max_number_of_basis_functions(basis_sets.auxillary)
  thead_integral_buffer = [zeros(Float64, max_primary_nbas^2*max_aux_nbas) for thread in 1:n_threads]
  if scf_options.load == "sequential"
    engine = jeri_engine_thread[1]
    integral_buffer = thead_integral_buffer[1]
    for cartesian_index in cartesian_indecies
      calculate_three_center_integrals_kernel!(three_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)
    end
  elseif scf_options.load == "static" || MPI.Comm_size(comm) == 1 || MPI.Comm_rank(comm) == 0
    Threads.@sync for batch_index in 1:batch_size+1:number_of_indecies
      Threads.@spawn begin
        do_three_center_integral_batch(batch_index,
         batch_size, cartesian_indecies, 
         three_center_integrals, jeri_engine_thread, 
         basis_sets, thead_integral_buffer, number_of_indecies)
      end
    end
  else
    # error("integral threading load type: $(scf_options.load) not supported")
  end
  three_center_integrals = MPI.Allreduce(three_center_integrals, MPI.SUM, MPI.COMM_WORLD)
  MPI.Barrier(comm)
  return three_center_integrals
end

@inline function do_three_center_integral_batch(batch_index, 
    batch_size, 
    cartesian_indecies, 
    three_center_integrals, 
    jeri_engine_thread, 
    basis_sets, 
    thead_integral_buffer,
    number_of_indecies)

    thread_index = Threads.threadid()
    engine = jeri_engine_thread[thread_index]
    buffer = thead_integral_buffer[thread_index]
    for view_index in batch_index:min(number_of_indecies, batch_index + batch_size)
      calculate_three_center_integrals_kernel!(three_center_integrals, 
        engine, 
        cartesian_indecies[view_index],
        basis_sets, 
        buffer)
    end
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
  
  copy_values_to_output!(three_center_integrals, integral_buffer, bf_1_pos, bf_2_pos, bf_3_pos, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis)
  axial_normalization_factor(three_center_integrals, shell_1, shell_2, shell_3, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis, bf_1_pos, bf_2_pos, bf_3_pos)
end

@inline function copy_values_to_output!(three_center_integrals, values, bf_1_pos, bf_2_pos, bf_3_pos, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis)
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
