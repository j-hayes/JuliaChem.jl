using Base.Threads
using LinearAlgebra
using TensorOperations 
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
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine, T2<:RHFTEIEngine }

  if scf_options.contraction_mode == "dense"
    @time df_rhf_fock_build_BLAS!(scf_data, jeri_engine_thread_df,
    basis_sets, occupied_orbital_coefficients, iteration, scf_options) 
  elseif scf_options.contraction_mode == "denseGPU"
    @time df_rhf_fock_build_dense_GPU!(scf_data, jeri_engine_thread_df, jeri_engine_thread,
    basis_sets, occupied_orbital_coefficients, iteration, scf_options)
 
  elseif scf_options.contraction_mode == "TensorOperationsGPU"
    df_rhf_fock_build_TensorOperations_GPU!(scf_data, jeri_engine_thread_df, jeri_engine_thread,
    basis_sets, occupied_orbital_coefficients, iteration, scf_options)  
  elseif scf_options.contraction_mode == "TensorOperations"
    # df_rhf_fock_build_TensorOperations!(scf_data, jeri_engine_thread_df, jeri_engine_thread,
    #   basis_sets, occupied_orbital_coefficients, iteration, scf_options)
    error("not implemented")
  elseif scf_options.contraction_mode == "GPU" # screened symmetric algorithm
    @time df_rhf_fock_build_GPU!(scf_data, jeri_engine_thread_df, jeri_engine_thread,
    basis_sets, occupied_orbital_coefficients, iteration, scf_options)
  else # default contraction mode is now scf_options.contraction_mode == "screened"
    @time df_rhf_fock_build_screened!(scf_data, jeri_engine_thread_df, jeri_engine_thread,
    basis_sets, occupied_orbital_coefficients, iteration, scf_options) 
  end

  comm = MPI.COMM_WORLD
  if MPI.Comm_size(comm) > 1
    MPI.Allreduce!(scf_data.two_electron_fock, MPI.SUM, comm)
  end
  return
end

function df_rhf_fock_build_BLAS!(scf_data, jeri_engine_thread_df::Vector{T}, basis_sets::CalculationBasisSets,
    occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}
  comm = MPI.COMM_WORLD
  indicies = get_df_static_basis_indices(basis_sets, MPI.Comm_size(comm), MPI.Comm_rank(comm))
  if iteration == 1
    two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
    three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options)
    calculate_D!(scf_data, two_center_integrals, three_center_integrals, basis_sets, indicies, scf_options)
    end  

  calculate_coulomb!(scf_data, occupied_orbital_coefficients , basis_sets, indicies,scf_options)
  calculate_exchange!(scf_data, occupied_orbital_coefficients ,basis_sets, indicies,scf_options)
end


function calculate_D!(scf_data, two_center_integrals, three_center_integrals, basis_sets, indicies, scf_options::SCFOptions)
  μμ = scf_data.μ
  νν = scf_data.μ
  AA = length(indicies)
  LinearAlgebra.LAPACK.potrf!('L', two_center_integrals)
  LinearAlgebra.LAPACK.trtri!('L', 'N', two_center_integrals)

  if MPI.Comm_size(MPI.COMM_WORLD) > 1
    two_center_integrals = two_center_integrals[:,indicies]
  end
  
  BLAS.gemm!('N', 'T', 1.0, reshape(three_center_integrals, (μμ*νν,scf_data.A)), two_center_integrals, 0.0, reshape(scf_data.D, (μμ*νν,AA)))
 
end

function calculate_coulomb!(scf_data, occupied_orbital_coefficients,  basis_sets, indicies, scf_options :: SCFOptions)
  AA = length(indicies)
  BLAS.gemm!('N', 'T', 1.0, occupied_orbital_coefficients, occupied_orbital_coefficients, 0.0, scf_data.density)
  

  BLAS.gemv!('T', 1.0, reshape(scf_data.D, (scf_data.μ*scf_data.μ,AA)), reshape(scf_data.density, scf_data.μ*scf_data.μ), 0.0, scf_data.coulomb_intermediate)
  BLAS.gemv!('N', 2.0, reshape(scf_data.D, (scf_data.μ*scf_data.μ,AA)), scf_data.coulomb_intermediate , 0.0, reshape(scf_data.two_electron_fock, scf_data.μ^2))

end

function calculate_exchange!(scf_data, occupied_orbital_coefficients,  basis_sets, indicies, scf_options :: SCFOptions)
  AA = length(indicies)
  μμ = scf_data.μ
  ii = scf_data.occ
  BLAS.gemm!('T', 'N' , 1.0, reshape(scf_data.D, (μμ, μμ*AA)), occupied_orbital_coefficients, 0.0, reshape(scf_data.D_tilde, (μμ*AA,ii)))
  BLAS.gemm!('N', 'T', -1.0, reshape(scf_data.D_tilde, (μμ, ii*AA)), reshape(scf_data.D_tilde, (μμ, ii*AA)), 1.0, scf_data.two_electron_fock)
end