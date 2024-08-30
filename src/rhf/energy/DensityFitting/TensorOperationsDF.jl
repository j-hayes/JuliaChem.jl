# using Base.Threads
# using LinearAlgebra
# using CUDA
# using TensorOperations 
# using JuliaChem.Shared.Constants.SCF_Keywords
# using JuliaChem.Shared
# using Serialization
# using HDF5
# using ThreadPinning

# function df_rhf_fock_build_TensorOperations!(scf_data, jeri_engine_thread_df::Vector{T}, jeri_engine_thread::Vector{T2},
#     basis_sets::CalculationBasisSets,
#     occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine,T2<:RHFTEIEngine}
#     comm = MPI.COMM_WORLD
#     if iteration == 1
#         two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
#         three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options)
#         J_AB_invt = convert(Array, inv(cholesky(Hermitian(two_center_integrals, :L)).U))
#         if MPI.Comm_size(comm) > 1
#             indicies = get_df_static_basis_indices(basis_sets, MPI.Comm_size(comm), MPI.Comm_rank(comm))
#             J_AB_invt = J_AB_invt[:, indicies]
#         end
#         @tensor scf_data.D[μ, ν, A] = three_center_integrals[μ, ν, BB] * J_AB_invt[BB, A]
#     end
#     # Coulomb
#     @tensor scf_data.density[μ, ν] = occupied_orbital_coefficients[μ, i] * occupied_orbital_coefficients[ν, i]
#     @tensor scf_data.coulomb_intermediate[A] = scf_data.D[μ, ν, A] * scf_data.density[μ, ν]
#     @tensor scf_data.two_electron_fock[μ, ν] = 2.0 * scf_data.coulomb_intermediate[A] * scf_data.D[μ, ν, A]

#     # Exchange
#     @tensor scf_data.D_tilde[ν, i, A] = scf_data.D[ν, μμ, A] * occupied_orbital_coefficients[μμ, i]
#     @tensor scf_data.two_electron_fock[μ, ν] -= scf_data.D_tilde[μ, i, A] * scf_data.D_tilde[ν, i, A]
# end


# function df_rhf_fock_build_GPU!(scf_data, jeri_engine_thread_df::Vector{T}, jeri_engine_thread ::Vector{T2},
#     basis_sets::CalculationBasisSets,
#     occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine, T2<:RHFTEIEngine}
#     comm = MPI.COMM_WORLD
  
    
#     if iteration == 1
  
#       two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
#       three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options)
#       cu_three_center_integrals =  CUDA.CuArray{Float64}(undef, (scf_data.μ, scf_data.μ, scf_data.A))
#       J_AB_invt = convert(Array, inv(cholesky(Hermitian(two_center_integrals, :L)).U))
  
  
      
#       cu_J_AB_invt =  CUDA.CuArray{Float64}(undef, (scf_data.A, scf_data.A))
  
#       copyto!(cu_three_center_integrals, three_center_integrals)
#       copyto!(cu_J_AB_invt, J_AB_invt)    
#       @tensor scf_data.D[μ, ν, A] = cu_three_center_integrals[μ, ν, BB] * cu_J_AB_invt[BB,A]
#       cu_three_center_integrals = nothing 
#       cu_J_AB_invt = nothing
#     end  
  
#     copyto!(scf_data.occupied_orbital_coefficients, occupied_orbital_coefficients)
  
#     @tensor scf_data.density[μ, ν] = scf_data.occupied_orbital_coefficients[μ,i]*scf_data.occupied_orbital_coefficients[ν,i]
#     @tensor scf_data.coulomb_intermediate[A] = scf_data.D[μ,ν,A]*scf_data.density[μ,ν]
#     @tensor scf_data.D_tilde[i,A,ν] = scf_data.D[ν, μμ, A]*scf_data.occupied_orbital_coefficients[μμ, i]
  
#     @tensor scf_data.two_electron_fock_GPU[μ, ν] = 2.0*scf_data.coulomb_intermediate[A]*scf_data.D[μ,ν,A]
#     @tensor scf_data.two_electron_fock_GPU[μ, ν] -= scf_data.D_tilde[i, A,μ]*scf_data.D_tilde[ i, A, ν]
    
#     copyto!(scf_data.two_electron_fock, scf_data.two_electron_fock_GPU)
  
  
#   end
  