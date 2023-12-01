using Base.Threads
using LinearAlgebra
using CUDA
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

function df_rhf_fock_build!(scf_data, jeri_engine_thread::Vector{T},
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}


  if scf_options.contraction_mode == "BLAS"
    df_rhf_fock_build_BLAS!(scf_data, jeri_engine_thread, 
      basis_sets, occupied_orbital_coefficients, iteration, scf_options)   
  elseif scf_options.contraction_mode == "BLAS_NoThread"
    df_rhf_fock_build_BLAS_NoThread!(scf_data, jeri_engine_thread, 
      basis_sets, occupied_orbital_coefficients, iteration, scf_options)    
  elseif scf_options.contraction_mode == "GPU"
    df_rhf_fock_build_GPU!(scf_data, jeri_engine_thread, 
    basis_sets, occupied_orbital_coefficients, iteration, scf_options)  
  else # default contraction mode is now TensorOperations
    df_rhf_fock_build_TensorOperations!(scf_data, jeri_engine_thread, 
      basis_sets, occupied_orbital_coefficients, iteration, scf_options)
  end

  comm = MPI.COMM_WORLD
  if MPI.Comm_size(comm) > 1
    MPI.Allreduce!(scf_data.two_electron_fock, MPI.SUM, comm)
  end
  
end

function df_rhf_fock_build_GPU!(scf_data, jeri_engine_thread::Vector{T},
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}
  comm = MPI.COMM_WORLD


  if iteration == 1
    three_center_integrals = calculate_three_center_integrals(jeri_engine_thread, basis_sets, scf_options)

    two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread, basis_sets, scf_options)

    cu_three_center_integrals =  CUDA.CuArray{Float64}(undef, (scf_data.μ, scf_data.μ, scf_data.A))
    J_AB_invt = convert(Array, inv(cholesky(Hermitian(two_center_integrals, :L)).U))


    
    cu_J_AB_invt =  CUDA.CuArray{Float64}(undef, (scf_data.A, scf_data.A))

    copyto!(cu_three_center_integrals, three_center_integrals)
    copyto!(cu_J_AB_invt, J_AB_invt)    
    @tensor scf_data.D[μ, ν, A] = cu_three_center_integrals[μ, ν, BB] * cu_J_AB_invt[BB,A]
    cu_three_center_integrals = nothing 
    cu_J_AB_invt = nothing
  end  

  copyto!(scf_data.occupied_orbital_coefficients, occupied_orbital_coefficients)

  @tensor scf_data.density[μ, ν] = scf_data.occupied_orbital_coefficients[μ,i]*scf_data.occupied_orbital_coefficients[ν,i]
  @tensor scf_data.coulomb_intermediate[A] = scf_data.D[μ,ν,A]*scf_data.density[μ,ν]
  @tensor scf_data.D_tilde[i,A,ν] = scf_data.D[ν, μμ, A]*scf_data.occupied_orbital_coefficients[μμ, i]

  @tensor scf_data.two_electron_fock_GPU[μ, ν] = 2.0*scf_data.coulomb_intermediate[A]*scf_data.D[μ,ν,A]
  @tensor scf_data.two_electron_fock_GPU[μ, ν] -= scf_data.D_tilde[i, A,μ]*scf_data.D_tilde[ i, A, ν]
  
  copyto!(scf_data.two_electron_fock, scf_data.two_electron_fock_GPU)


end

function df_rhf_fock_build_TensorOperations!(scf_data, jeri_engine_thread::Vector{T},
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}
  comm = MPI.COMM_WORLD
  if iteration == 1
    two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread, basis_sets, scf_options)
    three_center_integrals = calculate_three_center_integrals(jeri_engine_thread, basis_sets, scf_options)
    J_AB_invt = convert(Array, inv(cholesky(Hermitian(two_center_integrals, :L)).U))
    
    if MPI.Comm_size(comm) > 1
      indicies = get_df_static_basis_indices(basis_sets, MPI.Comm_size(comm), MPI.Comm_rank(comm))
      J_AB_invt = J_AB_invt[:,indicies]
    end
    @tensor scf_data.D[μ, ν, A] = three_center_integrals[μ, ν, BB] * J_AB_invt[BB,A]
  end  
  @tensor scf_data.density[μ, ν] = occupied_orbital_coefficients[μ,i]*occupied_orbital_coefficients[ν,i]
  @tensor scf_data.coulomb_intermediate[A] = scf_data.D[μ,ν,A]*scf_data.density[μ,ν]
  @tensor scf_data.two_electron_fock[μ, ν] = 2.0*scf_data.coulomb_intermediate[A]*scf_data.D[μ,ν,A]
  @tensor scf_data.D_tilde[ν,i,A] = scf_data.D[ν, μμ, A]*occupied_orbital_coefficients[μμ, i]
  @tensor scf_data.two_electron_fock[μ, ν] -= scf_data.D_tilde[μ, i, A]*scf_data.D_tilde[ν, i, A]
end

function df_rhf_fock_build_BLAS!(scf_data, jeri_engine_thread::Vector{T},
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}
  comm = MPI.COMM_WORLD
  indicies = get_df_static_basis_indices(basis_sets, MPI.Comm_size(comm), MPI.Comm_rank(comm))

  if iteration == 1
    two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread, basis_sets, scf_options)
    three_center_integrals = calculate_three_center_integrals(jeri_engine_thread, basis_sets, scf_options)
    calculate_D_BLAS!(scf_data, two_center_integrals, three_center_integrals, basis_sets, indicies, scf_options)
    scf_data.thread_two_electron_fock = [zeros(scf_data.μ, scf_data.μ) for i in 1:Threads.nthreads()]
  end  
  calculate_coulomb_BLAS!(scf_data, occupied_orbital_coefficients , basis_sets, indicies,scf_options)
  calculate_exchange_BLAS!(scf_data, occupied_orbital_coefficients ,basis_sets, indicies,scf_options)

end



function calculate_D_BLAS!(scf_data, two_center_integrals, three_center_integrals, basis_sets, indicies, scf_options::SCFOptions)
  μμ = scf_data.μ
  νν = scf_data.μ
  comm = MPI.COMM_WORLD
  J_AB_INV = convert(Array, inv(cholesky(Hermitian(two_center_integrals, :L)).U))
  if MPI.Comm_size(comm) > 1
    J_AB_INV = J_AB_INV[:,indicies]
  end
  Threads.@threads for ν in 1:μμ
    BLAS.gemm!('N', 'N', 1.0, view(three_center_integrals, :,ν,:), J_AB_INV, 0.0, view(scf_data.D, :,ν,:)) 
  end
  # BLAS.gemm!('N', 'N', 1.0, reshape(three_center_integrals, (μμ*νν,scf_data.A)), J_AB_INV, 0.0, reshape(scf_data.D, (μμ*νν,AA))) # old single gemm version
end


function calculate_coulomb_BLAS!(scf_data, occupied_orbital_coefficients,  basis_sets, indicies, scf_options :: SCFOptions)
  AA = length(indicies)


  BLAS.gemm!('N', 'T', 1.0, occupied_orbital_coefficients, occupied_orbital_coefficients, 0.0, scf_data.density)
  
  Threads.@threads for A in 1:AA
    BLAS.gemm!('N', 'N', 1.0, reshape(view(scf_data.D,:,:,A), (1,scf_data.μ*scf_data.μ)), reshape(scf_data.density, (scf_data.μ*scf_data.μ,1)), 0.0, view(scf_data.coulomb_intermediate, A:A))
  end

  Threads.@threads for ν in 1:scf_data.μ
    BLAS.gemm!('N', 'N', 2.0, view(scf_data.D, :,ν,:), scf_data.coulomb_intermediate, 0.0, view(scf_data.two_electron_fock, :,ν))
  end
  
  # BLAS.gemm!('N', 'N', 1.0,  reshape(scf_data.density, (1,scf_data.μ*scf_data.μ)), reshape(scf_data.D, (scf_data.μ*scf_data.μ,AA)),  0.0, reshape(scf_data.coulomb_intermediate, (1,AA))) old single blas call
  # BLAS.gemm!('N', 'N', 2.0, reshape(scf_data.D, (scf_data.μ*scf_data.μ, AA)), scf_data.coulomb_intermediate, 0.0, reshape(scf_data.two_electron_fock,(scf_data.μ*scf_data.μ, 1))) old single call blas
end

function calculate_exchange_BLAS!(scf_data, occupied_orbital_coefficients,  basis_sets, indicies, scf_options :: SCFOptions)
  AA = length(indicies)
  μμ = scf_data.μ
  ii = scf_data.occ
  n_threads = Threads.nthreads()

  Threads.@threads for A in 1:AA
    BLAS.gemm!('N', 'N', 1.0, view(scf_data.D,:,:, A), occupied_orbital_coefficients,0.0, view(scf_data.D_tilde, :, :,A))
  end  

  for thread in 1:n_threads
    scf_data.thread_two_electron_fock[thread] .= 0.0
  end  

  # @tensor scf_data.two_electron_fock[μ, ν] -= scf_data.D_tilde[μ, i, A]*scf_data.D_tilde[ν, i, A]

  Threads.@threads for thread in 1:n_threads
    for A in thread:n_threads:AA
      BLAS.gemm!('N', 'T', -1.0,  view(scf_data.D_tilde, :,:,A), view(scf_data.D_tilde, :,:,A),  1.0, scf_data.thread_two_electron_fock[thread])
    end
  end 
 
  for thread in 1:n_threads
    BLAS.axpy!(1.0, scf_data.thread_two_electron_fock[thread], scf_data.two_electron_fock)
  end  

  #old single call blas 
  # BLAS.gemm!('T', 'N' , 1.0, reshape(scf_data.D, (μμ, μμ*AA)), occupied_orbital_coefficients, 0.0, reshape(scf_data.D_tilde, (μμ*AA,ii)))
  # BLAS.gemm!('N', 'T', -1.0, reshape(scf_data.D_tilde, (μμ, ii*AA)), reshape(scf_data.D_tilde, (μμ, ii*AA)), 1.0, scf_data.two_electron_fock)
end


function df_rhf_fock_build_BLAS_NoThread!(scf_data, jeri_engine_thread::Vector{T},
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}
  comm = MPI.COMM_WORLD
  indicies = get_df_static_basis_indices(basis_sets, MPI.Comm_size(comm), MPI.Comm_rank(comm))
  if iteration == 1
    two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread, basis_sets, scf_options)
    three_center_integrals = calculate_three_center_integrals(jeri_engine_thread, basis_sets, scf_options)
    calculate_D_BLAS_NoThread!(scf_data, two_center_integrals, three_center_integrals, basis_sets, indicies, scf_options)
  end  
  calculate_coulomb_BLAS_NoThread!(scf_data, occupied_orbital_coefficients , basis_sets, indicies,scf_options)
  calculate_exchange_BLAS_NoThread!(scf_data, occupied_orbital_coefficients ,basis_sets, indicies,scf_options)

end

function calculate_D_BLAS_NoThread!(scf_data, two_center_integrals, three_center_integrals, basis_sets, indicies, scf_options::SCFOptions)
  μμ = scf_data.μ
  νν = scf_data.μ
  AA = length(indicies)
  J_AB_INV = convert(Array, inv(cholesky(Hermitian(two_center_integrals, :L)).U))
  if MPI.Comm_size(MPI.COMM_WORLD) > 1
    J_AB_INV = J_AB_INV[:,indicies]
  end
  BLAS.gemm!('N', 'N', 1.0, reshape(three_center_integrals, (μμ*νν,scf_data.A)), J_AB_INV, 0.0, reshape(scf_data.D, (μμ*νν,AA)))
 

end


function calculate_coulomb_BLAS_NoThread!(scf_data, occupied_orbital_coefficients,  basis_sets, indicies, scf_options :: SCFOptions)
  AA = length(indicies)
  # BLAS.symm!('L', 'L', 1.0, occupied_orbital_coefficients, occupied_orbital_coefficients, 0.0, scf_data.density)


  BLAS.gemm!('N', 'T', 1.0, occupied_orbital_coefficients, occupied_orbital_coefficients, 0.0, scf_data.density)

  # BLAS.gemm!('N', 'N', 1.0,  reshape(scf_data.density, (1,scf_data.μ*scf_data.μ)), reshape(scf_data.D, (scf_data.μ*scf_data.μ,AA)), 
  # 0.0, reshape(scf_data.coulomb_intermediate, (1,AA)))


  # denden = zeros(Float64, (scf_data.μ, scf_data.μ))
  # for i in 1:scf_data.μ
  #   for j in 1:scf_data.μ-1
  #       denden[i,j] = scf_data.density[i,j] * 2.0
  #       denden[j,i] = scf_data.density[j,i] * 2.0
  #   end
  #   denden[i,i] = scf_data.density[i,i]
  # end

  
  # #number of elements in the lower triangle of scf_data.density
  # lower_count = scf_data.μ*(scf_data.μ+1)÷2

  den = vec_triu_loop_double(scf_data.density)
  DD = zeros(Float64, (length(den), AA))
  # scf_data.D[1,2,1]
  for A in 1:AA
    DD[:,A] = vec_triu_loop(view(scf_data.D, :,:,A))
  end

  BLAS_number_of_threads = BLAS.get_num_threads()
  # BLAS.set_num_threads(1)
  # Threads.@threads for A in 1:AA
  #   BLAS.gemm!('T', 'N', 1.0,  den, view(DD, :,A), 0.0, view(scf_data.coulomb_intermediate, A:A))
  # end

  BLAS.gemm!('T', 'N', 1.0,  den, DD, 0.0, reshape(scf_data.coulomb_intermediate, (1,AA)))


  # BLAS.set_num_threads(BLAS_number_of_threads)

  BLAS.gemm!('N', 'N', 2.0, reshape(scf_data.D, (scf_data.μ*scf_data.μ, AA)), scf_data.coulomb_intermediate, 0.0,
    reshape(scf_data.two_electron_fock,(scf_data.μ*scf_data.μ, 1)))
end

function calculate_exchange_BLAS_NoThread!(scf_data, occupied_orbital_coefficients,  basis_sets, indicies, scf_options :: SCFOptions)
  AA = length(indicies)
  μμ = scf_data.μ
  ii = scf_data.occ
  BLAS.gemm!('T', 'N' , 1.0, reshape(scf_data.D, (μμ, μμ*AA)), occupied_orbital_coefficients, 0.0, reshape(scf_data.D_tilde, (μμ*AA,ii)))
  BLAS.gemm!('N', 'T', -1.0, reshape(scf_data.D_tilde, (μμ, ii*AA)), reshape(scf_data.D_tilde, (μμ, ii*AA)), 1.0, scf_data.two_electron_fock)
end
#https://discourse.julialang.org/t/half-vectorization/7399/4
function vech(A::AbstractMatrix{T}) where T
  m = LinearAlgebra.checksquare(A)
  v = Vector{T}(undef, (m*(m+1))>>1)
  k = 0
  for j = 1:m, i = j:m
      @inbounds v[k += 1] = A[i,j]
  end
  return v
end

# https://gist.github.com/tpapp/87e5675b87dbdd7a9a840e546eb20fae
function vec_triu_loop(M::AbstractMatrix{T}) where T
  m, n = size(M)
  m == n || throw(error("not square"))
  l = n*(n+1) ÷ 2
  v = Vector{T}(undef,l)
  k = 0
  @inbounds for i in 1:n
      for j in 1:i
          v[k + j] = M[j, i]
      end
      k += i
  end
  return v
end

function vec_triu_loop_double(M::AbstractMatrix{T}) where T
  m, n = size(M)
  m == n || throw(error("not square"))
  l = n*(n+1) ÷ 2
  v = Vector{T}(undef,l)
  k = 0
  @inbounds for i in 1:n
      for j in 1:i
          v[k + j] = M[j, i] * (i == j ? 1.0 : 2.0)
      end
      k += i
  end
  return v
end