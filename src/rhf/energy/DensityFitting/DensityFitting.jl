using Base.Threads
using LinearAlgebra
using TensorOperations
using JuliaChem.Shared.Constants.SCF_Keywords
using JuliaChem.Shared
#== Density Fitted Restricted Hartree-Fock, Fock build step ==#
#== 
indices for all tensor contractions 
  duplicated letters: (e.g.) dd dummy variables that will be summed Overlap
  A = Auxillary Basis orbital
  μ,ν = Primary Basis orbital
  i = occupied orbitals
==#

@inline function df_rhf_fock_build!(scf_data, jeri_engine_thread::Vector{T}, 
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}
  
  df_rhf_fock_build_multi_node!(scf_data, jeri_engine_thread::Vector{T}, 
  basis_sets::CalculationBasisSets,
  deepcopy(occupied_orbital_coefficients), iteration, scf_options::SCFOptions)
  comm = MPI.COMM_WORLD
  # if iteration == 1
  #   two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread, basis_sets, scf_options)
  #   three_center_integrals = calculate_three_center_integrals(jeri_engine_thread, basis_sets, scf_options)
  #   calculate_D!(scf_data, two_center_integrals, three_center_integrals, scf_options)
  # end
  # multi_thread_d_tilde = deepcopy(scf_data.D_tilde)
  # scf_data.D_tilde .= 0.0
  # calculate_D_tilde!(scf_data, occupied_orbital_coefficients, scf_options)

  # for index in CartesianIndices(scf_data.D_tilde)
  #   if !isapprox(multi_thread_d_tilde[index], scf_data.D_tilde[index])
  #     println(size(multi_thread_d_tilde))
  #     println(size(scf_data.D_tilde))
  #     println("multi_thread_d_tilde != scf_data.D_tilde in iteration $(iteration)")
  #     break 
  #   end
  # end

  # if iteration == 2
  #   println("D_tilde")
  #   for index in CartesianIndices(scf_data.D_tilde)
  #     println("scf_data.D_tilde[$(index[1]), $(index[2]), $(index[3])] = ", scf_data.D_tilde[index])
  #   end
  #   println("coulomb")
  #   for index in CartesianIndices(scf_data.two_electron_fock)
  #     println("scf_data.two_electron_fock[$(index[1]), $(index[2])] = ", scf_data.two_electron_fock[index])
  #   end
  # end

  coulomb_blas = deepcopy(scf_data.two_electron_fock)

  # scf_data.two_electron_fock .= 0.0
  # calculate_coulomb!(scf_data, occupied_orbital_coefficients, scf_options) 
  # # if iteration == 2 
  #   println("coulomb")
  #   for index in CartesianIndices(scf_data.two_electron_fock)
  #     if !isapprox(coulomb_blas[index], scf_data.two_electron_fock[index])
  #       println("coulomb_blas != scf_data.two_electron_fock in iteration $(iteration) @$(index)")
  #       break 
  #     end
  #   end
  # end
  calculate_exchange!(scf_data,scf_options)
  # if iteration == 2
  #   println("exchange")
  #   for index in CartesianIndices(scf_data.two_electron_fock)
  #     println("scf_data.exchange[$(index[1]), $(index[2])] = ", scf_data.two_electron_fock[index])
  #   end
  # end
end

@inline function df_rhf_fock_build_multi_node!(scf_data, jeri_engine_thread::Vector{T}, 
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}
  comm = MPI.COMM_WORLD

  if iteration == 1
    two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread, basis_sets, scf_options)
    three_center_integrals = calculate_three_center_integrals(jeri_engine_thread, basis_sets, scf_options)
    calculate_D_multi_thread!(scf_data, two_center_integrals, three_center_integrals, scf_options)
  end
  calculate_D_tilde_multi_thread!(scf_data, occupied_orbital_coefficients, scf_options)    
  calculate_coulomb_multi_thread!(scf_data, occupied_orbital_coefficients, scf_options)
  
  if MPI.Comm_rank(comm) == 0 && iteration == 2
    println("coulomb iteration 2")
    for index in CartesianIndices(scf_data.two_electron_fock)
      println("scf_data.two_electron_fock[$(index[1]), $(index[2]))] = ", scf_data.two_electron_fock[index])
    end
  end


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

function calculate_coulomb_multi_thread!(scf_data, occupied_orbital_coefficients, scf_options::SCFOptions)
  comm = MPI.COMM_WORLD
  scf_data.two_electron_fock .= 0.0
  for A in 1:scf_data.A
    scf_data.coulomb_intermediate[A] = BLAS.dot(scf_data.D_tilde[:, A, :], convert(Array, occupied_orbital_coefficients))
  end

  for i in 1:scf_data.μ
    scf_data.two_electron_fock[i, :] = BLAS.gemm('N', 'N', 2.0, scf_data.D[i, :, :], scf_data.coulomb_intermediate)
  end
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

# calculate xiK 
@inline function calculate_D_tilde_multi_thread!(scf_data, occupied_orbital_coefficients, scf_options::SCFOptions)
  comm = MPI.COMM_WORLD
  scf_data.D_tilde = zeros(Float64, (scf_data.μ, scf_data.A, scf_data.occ))
  scf_data.D = reshape(scf_data.D, (scf_data.μ, scf_data.μ, scf_data.A))

  batch_size = convert(Int,ceil(scf_data.μ / MPI.Comm_size(comm)))

  batch_start = batch_size * MPI.Comm_rank(comm) + 1
  batch_end = min(batch_size + batch_start, scf_data.μ)

  @sync for μ in batch_start:batch_end
    Threads.@spawn begin
      scf_data.D_tilde[μ,:,:] =  BLAS.gemm('T', 'N' , 1.0, scf_data.D[:, μ, :], occupied_orbital_coefficients)
    end
  end 

    
  MPI.Barrier(comm)
  scf_data.D_tilde .= MPI.Allreduce(scf_data.D_tilde, MPI.SUM, MPI.COMM_WORLD)
  MPI.Barrier(comm)

end


@inline function calculate_D!(scf_data, two_center_integrals, three_center_integrals, scf_options::SCFOptions)
  comm = MPI.COMM_WORLD
  # this needs to be mpi parallelized
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



function calculate_D_multi_thread!(scf_data, two_center_integrals, three_center_integrals, scf_options::SCFOptions)
  comm = MPI.COMM_WORLD
  # this needs to be mpi parallelized
  J_AB_invt = convert(Array, transpose(cholesky(Hermitian(two_center_integrals, :L)).L \I))

  μμ = scf_data.μ
  νν = scf_data.μ
  AA = scf_data.A
  # use transpose transpose gemm transpose to perform tensor contraction 
  # Linv_T is already a 2D matrix so no need to reshape, and is in correct order
  # scf_data.D = reshape(scf_data.D, (μμ*νν,AA))
  # three_center_integrals =  reshape(three_center_integrals, (μμ*νν,AA))

  batch_size = convert(Int,ceil(scf_data.μ / MPI.Comm_size(comm)))

  batch_start = batch_size * MPI.Comm_rank(comm) + 1
  batch_end = min(batch_size + batch_start, scf_data.μ)

  @sync for μ in batch_start:batch_end
    Threads.@spawn begin
        scf_data.D[μ,:,:] = BLAS.gemm('N', 'N' , three_center_integrals[μ,:,:], J_AB_invt)
    end
  end 
  
  MPI.Barrier(comm)
  scf_data.D .= MPI.Allreduce(scf_data.D, MPI.SUM, MPI.COMM_WORLD)
  MPI.Barrier(comm)

  # permutedims!(scf_data.D_permuted, scf_data.D, (1,3,2))
  # scf_data.D_permuted = reshape(scf_data.D_permuted, (νν*AA, μμ))
end

