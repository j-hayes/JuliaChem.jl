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
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions, H::Array{Float64}) where {T<:DFRHFTEIEngine, T2<:RHFTEIEngine }

  comm = MPI.COMM_WORLD
  rank = MPI.Comm_rank(comm)

  
 
  if scf_options.contraction_mode == "GPU"  # screened symmetric algorithm
    #get environment variable for use dense 
    use_dense = false
    if haskey(ENV, "JC_USE_DENSE")
      use_dense = parse(Bool, ENV["JC_USE_DENSE"])
      if iteration == 1
        println("use dense = $use_dense")
      end
    end
    use_adaptive = false
    num_devices = 1
    if haskey(ENV, "JC_USE_ADAPTIVE")
      use_adaptive = parse(Bool, ENV["JC_USE_ADAPTIVE"])
      if iteration == 1
        println("use adaptive = $use_adaptive")
      end
    end
    if haskey(ENV, "JC_NUM_DEVICES")
      num_devices = parse(Int64, ENV["JC_NUM_DEVICES"])
    end

    if use_dense || num_devices == 1 && use_adaptive && scf_data.μ < 800 && rank == 0 && MPI.Comm_size(comm) == 1 # used for small systems on runs with a single rank only uses one device
        df_rhf_fock_build_dense_GPU!(scf_data, jeri_engine_thread_df, jeri_engine_thread,
        basis_sets, occupied_orbital_coefficients, iteration, scf_options, H)
    else
      df_rhf_fock_build_GPU!(scf_data, jeri_engine_thread_df, jeri_engine_thread,
      basis_sets, occupied_orbital_coefficients, iteration, scf_options, H)
    end    
  else
    if scf_options.contraction_mode == "dense"
      df_rhf_fock_build_BLAS!(scf_data, jeri_engine_thread_df,
      basis_sets, occupied_orbital_coefficients, iteration, scf_options) 
    end
    
    #default contraction mode is now scf_options.contraction_mode == "screened"
    df_rhf_fock_build_screened!(scf_data, jeri_engine_thread_df, jeri_engine_thread,
    basis_sets, occupied_orbital_coefficients, iteration, scf_options) 
   
    if rank == 0
      scf_data.two_electron_fock .+= H # add the core hamiltonian to the two electron fock matrix
    end
  end

 

  if MPI.Comm_size(comm) > 1
    MPI.Allreduce!(scf_data.two_electron_fock, MPI.SUM, comm)
  end  
  return scf_data.two_electron_fock
end

function allocate_memory_density_fitting_dense(scf_data, indicies)
  AA = length(indicies)
  μμ = scf_data.μ
  ii = scf_data.occ
  scf_data.D = zeros(Float64, (μμ, μμ, AA))
  scf_data.D_tilde = zeros(Float64, (μμ, AA,ii))
  scf_data.coulomb_intermediate = zeros(Float64, AA)
  scf_data.two_electron_fock = zeros(Float64, (μμ, μμ))
  scf_data.density = zeros(Float64, (μμ, μμ))

  return
end

function df_rhf_fock_build_BLAS!(scf_data, jeri_engine_thread_df::Vector{T}, basis_sets::CalculationBasisSets,
    occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}
  comm = MPI.COMM_WORLD
  indicies = get_df_static_basis_indices(basis_sets, MPI.Comm_size(comm), MPI.Comm_rank(comm))
  if iteration == 1
    allocate_memory_density_fitting_dense(scf_data, indicies)
    two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
    three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options)
    calculate_D!(scf_data, two_center_integrals, three_center_integrals, basis_sets, indicies, scf_options)
  end  
  calculate_exchange!(scf_data, occupied_orbital_coefficients ,basis_sets, indicies,scf_options)
  calculate_coulomb!(scf_data, occupied_orbital_coefficients , basis_sets, indicies,scf_options)
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

  scf_data.D = zeros(size(three_center_integrals))
  
  BLAS.gemm!('N', 'T', 1.0, reshape(three_center_integrals, (μμ*νν,scf_data.A)), two_center_integrals, 0.0, reshape(scf_data.D, (μμ*νν,AA)))
 
end

function calculate_coulomb!(scf_data, occupied_orbital_coefficients,  basis_sets, indicies, scf_options :: SCFOptions)
  AA = length(indicies)
  BLAS.gemm!('N', 'T', 1.0, occupied_orbital_coefficients, occupied_orbital_coefficients, 0.0, scf_data.density)
  

  BLAS.gemv!('T', 1.0, reshape(scf_data.D, (scf_data.μ*scf_data.μ,AA)), reshape(scf_data.density, scf_data.μ*scf_data.μ), 0.0, scf_data.coulomb_intermediate)
  BLAS.gemv!('N', 2.0, reshape(scf_data.D, (scf_data.μ*scf_data.μ,AA)), scf_data.coulomb_intermediate , 1.0, reshape(scf_data.two_electron_fock, scf_data.μ^2))

end

function calculate_exchange!(scf_data, occupied_orbital_coefficients,  basis_sets, indicies, scf_options :: SCFOptions)
  AA = length(indicies)
  μμ = scf_data.μ
  ii = scf_data.occ
  BLAS.gemm!('T', 'N' , 1.0, reshape(scf_data.D, (μμ, μμ*AA)), occupied_orbital_coefficients, 0.0, reshape(scf_data.D_tilde, (μμ*AA,ii)))
  BLAS.gemm!('N', 'T', -1.0, reshape(scf_data.D_tilde, (μμ, ii*AA)), reshape(scf_data.D_tilde, (μμ, ii*AA)), 0.0, scf_data.two_electron_fock)
end