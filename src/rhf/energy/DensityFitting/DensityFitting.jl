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
  if scf_options.contraction_mode == "TensorOperations"
    df_rhf_fock_build_TensorOperations!(scf_data, jeri_engine_thread, 
      basis_sets, occupied_orbital_coefficients, iteration, scf_options)
  else
    df_rhf_fock_build_BLAS!(scf_data, jeri_engine_thread, 
      basis_sets, occupied_orbital_coefficients, iteration, scf_options)
  end
end

@inline function df_rhf_fock_build_TensorOperations!(scf_data, jeri_engine_thread::Vector{T},
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}
  
  if iteration == 1
    two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread, basis_sets, scf_options)
    three_center_integrals = calculate_three_center_integrals(jeri_engine_thread, basis_sets, scf_options)

    J_AB_invt = convert(Array, transpose(cholesky(Hermitian(two_center_integrals, :L)).L \ I))
    indicies = get_df_static_basis_indices(basis_sets, MPI.Comm_size(comm), MPI.Comm_rank(comm))
    @tensor scf_data.D[μ, ν, A] = three_center_integrals[μ, ν, BB] * J_AB_invt[:, indicies][BB,A]

  end  
  @tensor scf_data.density[μ, ν] := occupied_orbital_coefficients[μ,i]*occupied_orbital_coefficients[ν,i]
  @tensor scf_data.coulomb_intermediate[A] = scf_data.D[μ,ν,A]*density[μ,ν]
  @tensor scf_data.two_electron_fock[μ, ν] = 2.0*scf_data.coulomb_intermediate[A]*scf_data.D[μ,ν,A]
  @tensor scf_data.D_tilde[ν,i,A] = scf_data.D[ν, μμ, A]*occupied_orbital_coefficients[μμ, i]
  @tensor scf_data.two_electron_fock[μ, ν] -= scf_data.D_tilde[μ, i, A]*scf_data.D_tilde[ν, i, A]
  
  MPI.Barrier(comm)
  MPI.Allreduce!(scf_data.two_electron_fock, MPI.SUM, comm)
  MPI.Barrier(comm)
end

@inline function df_rhf_fock_build_BLAS!(scf_data, jeri_engine_thread::Vector{T},
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}
  comm = MPI.COMM_WORLD
  indicies = get_df_static_basis_indices(basis_sets, MPI.Comm_size(comm), MPI.Comm_rank(comm))

  if iteration == 1
    println("doing BLAS DF algorithm")

    two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread, basis_sets, scf_options)
    three_center_integrals = calculate_three_center_integrals(jeri_engine_thread, basis_sets, scf_options)
    calculate_D_BLAS!(scf_data, two_center_integrals, three_center_integrals, basis_sets, indicies, scf_options)
  end  
  
  @tensor scf_data.density[μ, ν] = occupied_orbital_coefficients[μ,i]*occupied_orbital_coefficients[ν,i]
  calculate_coulomb_BLAS!(scf_data, occupied_orbital_coefficients ,basis_sets, indicies,scf_options)
  calculate_exchange_BLAS!(scf_data, occupied_orbital_coefficients ,basis_sets, indicies,scf_options)
  
  
  MPI.Barrier(comm)
  MPI.Allreduce!(scf_data.two_electron_fock, MPI.SUM, comm)
  MPI.Barrier(comm)
end

@inline function calculate_D_BLAS!(scf_data, two_center_integrals, three_center_integrals, basis_sets, indicies, scf_options::SCFOptions)
  comm = MPI.COMM_WORLD
  # this needs to be mpi parallelized
  J_AB_invt = convert(Array, transpose(cholesky(Hermitian(two_center_integrals, :L)).L \I))

  ## todo get this from a parameter to make this more readable 
  μμ = scf_data.μ
  νν = scf_data.μ
  AA = length(indicies)

  # use transpose transpose gemm transpose to perform tensor contraction 
  # J_AB_invt is already a 2D matrix so no need to reshape, and is in correct order
  BLAS.gemm!('N', 'N', 1.0, reshape(three_center_integrals, (μμ*νν,scf_data.A)), J_AB_invt[:, indicies], 0.0, reshape(scf_data.D, (μμ*νν,AA)))
  # MPI.Barrier(comm)
end # end function calculate_D

@inline function calculate_coulomb_BLAS!(scf_data, occupied_orbital_coefficients,  basis_sets, indicies, scf_options :: SCFOptions)
  AA = length(indicies)
  scf_data.coulomb_intermediate = zeros(Float64, (AA,1))
  @time @tensor density[μ, ν] := occupied_orbital_coefficients[μ,i]*occupied_orbital_coefficients[ν,i]
  @time begin 
    for aux_index in eachindex(indicies)
    scf_data.coulomb_intermediate[aux_index] = dot(scf_data.D[:,:,aux_index], density[:,:])  # @tensor scf_data.coulomb_intermediate[A] = scf_data.D[μ,ν,A]*density[μ,ν]
    end
  end

  @time BLAS.gemm!('N', 'N', 2.0, reshape(scf_data.D, (scf_data.μ*scf_data.μ, AA)), scf_data.coulomb_intermediate, 0.0,
    reshape(scf_data.two_electron_fock,(scf_data.μ*scf_data.μ, 1)))
end

@inline function calculate_exchange_BLAS!(scf_data, occupied_orbital_coefficients,  basis_sets, indicies, scf_options :: SCFOptions)
  AA = length(indicies)
  μμ = scf_data.μ
  oo = scf_data.occ # number of occupied orbitals
  ii = scf_data.occ
  
  BLAS.gemm!('T', 'N' , 1.0, reshape(scf_data.D, (μμ, μμ*AA)), occupied_orbital_coefficients, 0.0, reshape(scf_data.D_tilde, (μμ*AA,ii)))
  BLAS.gemm!('N', 'T', -1.0, reshape(scf_data.D_tilde, (μμ, oo*AA)), reshape(scf_data.D_tilde, (μμ, oo*AA)), 1.0, scf_data.two_electron_fock)

end