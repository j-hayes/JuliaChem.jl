using Base.Threads
using LinearAlgebra
using TensorOperations


#== Density Fitted Restricted Hartree-Fock, Fock build step ==#
function df_rhf_fock_build(engine::DFRHFTEIEngine, basis_sets::CalculationBasisSets, occupied_orbital_coefficients) 
    
  three_center_integrals = calculate_three_center_integrals(engine, basis_sets)
  two_center_integrals = calculate_two_center_intgrals(engine, basis_sets)
  two_electron_fock_component = contract_two_electron_integrals(three_center_integrals, two_center_integrals, occupied_orbital_coefficients, basis_sets)

  return two_electron_fock_component
end

@inline function calculate_three_center_integrals(engine::DFRHFTEIEngine, basis_sets) :: Array{Float64}
  
  auxilliary_basis_shell_count = length(basis_sets.auxillary)
  basis_shell_count = length(basis_sets.primary)
  auxillary_basis_function_count =  basis_sets.auxillary.norb
  basis_function_count =  basis_sets.primary.norb
  three_center_integrals = Array{Float64}(undef, (auxillary_basis_function_count,basis_function_count,basis_function_count))
  
  for s1 in 1:auxilliary_basis_shell_count
    shell_1 = basis_sets.auxillary.shells[s1]
    shell_1_nbasis = shell_1.nbas
    bf_1_pos = shell_1.pos

    for s2 in 1:basis_shell_count
      shell_2 = basis_sets.primary.shells[s2]
      shell_2_nbasis = shell_2.nbas
      bf_2_pos = shell_2.pos
      n12 = shell_1_nbasis * shell_2_nbasis

      for s3 in 1:basis_shell_count
        shell_3 = basis_sets.primary.shells[s3]
        shell_3_nbasis = shell_3.nbas
        bf_3_pos = shell_3.pos

        n123 = n12 * shell_3_nbasis
        integral_values = Vector{Float64}(undef, n123)
        JERI.compute_eri_block_df(engine, integral_values, s1, s2, s3, n123, 0)      

        copy_values_to_output!(three_center_integrals, integral_values, bf_1_pos, bf_2_pos, bf_3_pos, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis)
        axial_normalization_factor(three_center_integrals, shell_1, shell_2, shell_3, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis, bf_1_pos, bf_2_pos, bf_3_pos)
      end
    end
  end 

  return three_center_integrals
end

@inline function copy_values_to_output!(three_center_integrals, values, bf_1_pos, bf_2_pos, bf_3_pos, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis)        
  temp_index = 1
  for i in bf_1_pos:bf_1_pos+shell_1_nbasis-1
    for j in bf_2_pos:bf_2_pos+shell_2_nbasis-1
      for k in bf_3_pos:bf_3_pos+shell_3_nbasis-1
        three_center_integrals[i,j,k] = values[temp_index]
        temp_index += 1
      end
    end
  end
end

@inline function calculate_two_center_intgrals(engine::DFRHFTEIEngine, basis_sets) :: Matrix{Float64}
  auxiliary_basis_function_count = basis_sets.auxillary.norb
  auxilliary_basis_shell_count = length(basis_sets.auxillary)
  two_center_integrals = zeros((auxiliary_basis_function_count, auxiliary_basis_function_count))

  for shell_1_index in 1:auxilliary_basis_shell_count
    shell_1 = basis_sets.auxillary.shells[shell_1_index]
    shell_1_basis_count = shell_1.nbas
    bf_1_pos = shell_1.pos

    for shell_2_index in 1:shell_1_index
      shell_2 = basis_sets.auxillary.shells[shell_2_index]
      shell_2_basis_count = shell_2.nbas
      bf_2_pos = shell_2.pos

      integral_values = Vector{Float64}(undef, shell_1_basis_count*shell_2_basis_count)
      JERI.compute_two_center_eri_block(engine, integral_values, shell_1_index-1, shell_2_index-1, shell_1_basis_count, shell_2_basis_count)     
      copy_values_to_output!(two_center_integrals, integral_values, shell_1, shell_2, shell_1_basis_count, shell_2_basis_count)
      axial_normalization_factor(two_center_integrals, shell_1, shell_2, shell_1_basis_count, shell_2_basis_count, bf_1_pos, bf_2_pos)
    end 
  end
  clear_matrix_upper_triangle!(two_center_integrals, auxiliary_basis_function_count)
  return two_center_integrals
end

@inline function copy_values_to_output!(two_center_integrals, values, shell_1, shell_2, shell_1_basis_count, shell_2_basis_count)
  temp_index = 1
  for i in  shell_1.pos:shell_1.pos+shell_1_basis_count-1
      for j in shell_2.pos:shell_2.pos+shell_2_basis_count-1         
        two_center_integrals[i,j] = values[temp_index]
        temp_index += 1
      end 
  end
end

@inline function clear_matrix_upper_triangle!(two_center_integrals, auxiliary_basis_function_count)
  for i in 1:auxiliary_basis_function_count
    for j in i+1:auxiliary_basis_function_count 
        two_center_integrals[i,j] = 0.0
    end
  end
end

@inline function contract_two_electron_integrals(Zxy_Matrix, eri_block_2_center_matrix, occupied_orbital_coefficients, basis_sets)

  aux_basis_function_count = basis_sets.auxillary.norb
  basis_function_count = basis_sets.primary.norb

  hermitian_eri_block_2_center_matrix = Hermitian(eri_block_2_center_matrix, :L)
  LLT_2_center = cholesky(hermitian_eri_block_2_center_matrix)
  two_center_cholesky_lower = LLT_2_center.L
  
  Linv_t = convert(Array, transpose(two_center_cholesky_lower \I))
  number_off_occ_orbitals = size(occupied_orbital_coefficients, 2)
  xyK = zeros(basis_function_count, basis_function_count, aux_basis_function_count)
  TensorOperations.tensorcontract!(1.0, Zxy_Matrix, (2, 3, 4), 'N',  Linv_t, (2, 5), 'N', 0.0, xyK, (3, 4, 5))
  xiK = zeros(basis_function_count, number_off_occ_orbitals, aux_basis_function_count)
  TensorOperations.tensorcontract!(1.0, xyK, (1,2,3), 'N',  occupied_orbital_coefficients, (2,4), 'N', 0.0, xiK, (1,4,3))

  two_electron_fock_component = zeros(basis_function_count,basis_function_count)

  TensorOperations.tensorcontract!(1.0, xiK, (1,2,3), 'N',  xiK, (4, 2, 3), 'N', 0.0, two_electron_fock_component, (1, 4))
  Jtmp = zeros(aux_basis_function_count)
  TensorOperations.tensorcontract!(1.0, xiK, (1, 2, 3), 'N', occupied_orbital_coefficients, (1, 2),  'N',  0.0, Jtmp, (3))
  TensorOperations.tensorcontract!(2.0, xyK, (1, 2, 3), 'N',  Jtmp, (3), 'N', -1.0, two_electron_fock_component, (1, 2))

  return two_electron_fock_component
end