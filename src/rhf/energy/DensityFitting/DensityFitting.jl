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
  if scf_options.contraction_mode == "BLAS"
    df_rhf_fock_build_BLAS!(scf_data, jeri_engine_thread, 
      basis_sets, occupied_orbital_coefficients, iteration, scf_options)    
  else # default contraction mode is now TensorOperations
    df_rhf_fock_build_TensorOperations!(scf_data, jeri_engine_thread, 
      basis_sets, occupied_orbital_coefficients, iteration, scf_options)
  end
end

@inline function df_rhf_fock_build_TensorOperations!(scf_data, jeri_engine_thread::Vector{T},
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}
  comm = MPI.COMM_WORLD
  if iteration == 1
    two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread, basis_sets, scf_options)
    three_center_integrals = calculate_three_center_integrals(jeri_engine_thread, basis_sets, scf_options)
    J_AB_invt = inv(cholesky(Hermitian(two_center_integrals, :L)).L)
    indicies = get_df_static_basis_indices(basis_sets, MPI.Comm_size(comm), MPI.Comm_rank(comm))
    @tensor scf_data.D[μ, ν, A] = three_center_integrals[μ, ν, BB] * J_AB_invt[indicies,:][A,BB]

  end  
  GC.safepoint()

  @tensor scf_data.density[μ, ν] = occupied_orbital_coefficients[μ,i]*occupied_orbital_coefficients[ν,i]
  @tensor scf_data.coulomb_intermediate[A] = scf_data.D[μ,ν,A]*scf_data.density[μ,ν]
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
    two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread, basis_sets, scf_options)
    three_center_integrals = calculate_three_center_integrals(jeri_engine_thread, basis_sets, scf_options)
    calculate_D_BLAS!(scf_data, two_center_integrals, three_center_integrals, basis_sets, indicies, scf_options)
  end  
  calculate_coulomb_BLAS!(scf_data, occupied_orbital_coefficients , basis_sets, indicies,scf_options)
  calculate_exchange_BLAS!(scf_data, occupied_orbital_coefficients ,basis_sets, indicies,scf_options)

  MPI.Barrier(comm)
  MPI.Allreduce!(scf_data.two_electron_fock, MPI.SUM, comm)
  MPI.Barrier(comm)
end

@inline function calculate_D_BLAS!(scf_data, two_center_integrals, three_center_integrals, basis_sets, indicies, scf_options::SCFOptions)
  comm = MPI.COMM_WORLD
  μμ = scf_data.μ
  νν = scf_data.μ
  AA = length(indicies)

  LAPACK.potrf!('L', two_center_integrals)
  J_AB_INV = inv(two_center_integrals)[ indicies,:]
  BLAS.gemm!('N', 'T', 1.0, reshape(three_center_integrals, (μμ*νν,scf_data.A)), J_AB_INV, 0.0, reshape(scf_data.D, (μμ*νν,AA)))
  scf_data.D = reshape(scf_data.D, (μμ, νν,AA))
  # MPI.Barrier(comm)
  # MPI.Allreduce!(scf_data.D, MPI.SUM, comm)
  # MPI.Barrier(comm)

  # if MPI.Comm_rank(comm) == 0
  #    io = open("D_2proc.txt", "w")
  #    for index in CartesianIndices(scf_data.D)
  #      write(io, "D[$(index[1]), $(index[2]), $(index[3])] = $(scf_data.D[index])\n")
  #    end
  # end


end # end function calculate_D


@inline function calculate_coulomb_BLAS!(scf_data, occupied_orbital_coefficients,  basis_sets, indicies, scf_options :: SCFOptions)
  AA = length(indicies)
  BLAS.gemm!('N', 'T', 1.0, occupied_orbital_coefficients, occupied_orbital_coefficients, 0.0, scf_data.density)
  BLAS.gemm!('N', 'N', 1.0,  reshape(scf_data.density, (1,scf_data.μ*scf_data.μ)), reshape(scf_data.D, (scf_data.μ*scf_data.μ,AA)), 
    0.0, reshape(scf_data.coulomb_intermediate, (1,AA)))
  BLAS.gemm!('N', 'N', 2.0, reshape(scf_data.D, (scf_data.μ*scf_data.μ, AA)), scf_data.coulomb_intermediate, 0.0,
    reshape(scf_data.two_electron_fock,(scf_data.μ*scf_data.μ, 1)))
end

@inline function calculate_exchange_BLAS!(scf_data, occupied_orbital_coefficients,  basis_sets, indicies, scf_options :: SCFOptions)
  AA = length(indicies)
  μμ = scf_data.μ
  ii = scf_data.occ
  
  BLAS.gemm!('T', 'N' , 1.0, reshape(scf_data.D, (μμ, μμ*AA)), occupied_orbital_coefficients, 0.0, reshape(scf_data.D_tilde, (μμ*AA,ii)))
  BLAS.gemm!('N', 'T', -1.0, reshape(scf_data.D_tilde, (μμ, ii*AA)), reshape(scf_data.D_tilde, (μμ, ii*AA)), 1.0, scf_data.two_electron_fock)

end