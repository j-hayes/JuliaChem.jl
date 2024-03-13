using Base.Threads
using LinearAlgebra
using CUDA
using TensorOperations 
using JuliaChem.Shared.Constants.SCF_Keywords
using JuliaChem.Shared
using Serialization
using HDF5

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

  
  if scf_options.contraction_mode == "screened"
    df_rhf_fock_build_screened!(scf_data, jeri_engine_thread_df, jeri_engine_thread,
      basis_sets, occupied_orbital_coefficients, iteration, scf_options)
  elseif scf_options.contraction_mode == "GPU"
    df_rhf_fock_build_GPU!(scf_data, jeri_engine_thread_df, jeri_engine_thread,
    basis_sets, occupied_orbital_coefficients, iteration, scf_options)  
  elseif scf_options.contraction_mode == "TensorOperations"
    df_rhf_fock_build_TensorOperations!(scf_data, jeri_engine_thread_df, jeri_engine_thread,
      basis_sets, occupied_orbital_coefficients, iteration, scf_options)
  else # default contraction mode is now scf_options.contraction_mode == "BLAS_NoThread"
      df_rhf_fock_build_BLAS!(scf_data, jeri_engine_thread_df,
        basis_sets, occupied_orbital_coefficients, iteration, scf_options)  
  end

  comm = MPI.COMM_WORLD
  if MPI.Comm_size(comm) > 1
    MPI.Allreduce!(scf_data.two_electron_fock, MPI.SUM, comm)
  end
  
end

function df_rhf_fock_build_GPU!(scf_data, jeri_engine_thread_df::Vector{T}, jeri_engine_thread ::Vector{T2},
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine, T2<:RHFTEIEngine}
  comm = MPI.COMM_WORLD

  
  if iteration == 1

    two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
    three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options)
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

function df_rhf_fock_build_TensorOperations!(scf_data, jeri_engine_thread_df::Vector{T}, jeri_engine_thread ::Vector{T2},
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine, T2<:RHFTEIEngine}
  comm = MPI.COMM_WORLD
  if iteration == 1
    two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
    
    three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options)
    three_eri_triangle = zeros(Float64, (scf_data.μ*(scf_data.μ+1)÷2, scf_data.A))
    Threads.@threads for A in 1:scf_data.A
      three_eri_triangle[:,A] = vec_triu_loop(view(three_center_integrals, :,:,A))
    end
    J_AB_invt = convert(Array, inv(cholesky(Hermitian(two_center_integrals, :L)).U))
    
    if MPI.Comm_size(comm) > 1
      indicies = get_df_static_basis_indices(basis_sets, MPI.Comm_size(comm), MPI.Comm_rank(comm))
      J_AB_invt = J_AB_invt[:,indicies]
    end
    @tensor scf_data.D[μ, ν, A] = three_center_integrals[μ, ν, BB] * J_AB_invt[BB,A]
    @tensor scf_data.D_triangle[μ, A] = three_eri_triangle[μ, BB] * J_AB_invt[BB,A]
  end  

  @tensor scf_data.density[μ, ν] = occupied_orbital_coefficients[μ,i]*occupied_orbital_coefficients[ν,i]
  den = vec_triu_loop_double(scf_data.density)
  
  # println("coulmb regular")
  # begin
  @tensor scf_data.coulomb_intermediate[A] = scf_data.D[μ,ν,A]*scf_data.density[μ,ν]
  @tensor scf_data.two_electron_fock[μ,ν] = 2.0*scf_data.coulomb_intermediate[A]*scf_data.D[μ,ν,A]
  # end
  # println("coulomb sym")
  # begin
    @tensor scf_data.coulomb_intermediate[A] = scf_data.D_triangle[μ,A]*den[μ]
    @tensor scf_data.two_electron_fock_triangle[μ] = 2.0*scf_data.coulomb_intermediate[A]*scf_data.D_triangle[μ,A]

    # begin 
      Threads.@threads for i in 1:scf_data.μ
        ii = i*(i-1)÷2 + 1
        for j in 1:i-1
          scf_data.two_electron_fock[i,j] = scf_data.two_electron_fock_triangle[ii]
          scf_data.two_electron_fock[j,i] = scf_data.two_electron_fock_triangle[ii]
          ii += 1
        end
        ii = i*(i-1)÷2 + i
        scf_data.two_electron_fock[i,i] = scf_data.two_electron_fock_triangle[ii]
      end
    # end
  # end
  # d_tilde = zeros(size(scf_data.D_tilde))
  d_tilde = zeros(Float64, (scf_data.occ*scf_data.A, scf_data.μ))

  for p in 1:scf_data.μ
    for i in 1:scf_data.occ
      for A in 1:scf_data.A
        index = i*scf_data.occ + A
        start_triangle_index = p*(p-1)÷2 + 1
        end_triangle_index = start_triangle_index + scf_data.μ 
        d_tilde[index,p] = (occupied_orbital_coefficients[:,i]*view(scf_data.D_triangle, p:start_triangle_index,A))[1]
      end
    end
    @tensor d_tilde[iQ, p] = occupied_orbital_coefficients[μμ, i]*view(scf_data.D_triangle)[μμ, p]

  end
  # @tensor d_tilde[i,ν] = scf_data.D[ν, μμ, A]*occupied_orbital_coefficients[μμ, i]
  @tensor scf_data.D_tilde[ν,i,A] = scf_data.D[ν, μμ, A]*occupied_orbital_coefficients[μμ, i]
  @tensor scf_data.two_electron_fock[μ, ν] -=  scf_data.D_tilde[μ,i,A]*scf_data.D_tilde[ν, i, A]



end



function df_rhf_fock_build_BLAS_NoThread_sym!(scf_data, jeri_engine_thread_df::Vector{T}, jeri_engine_thread ::Vector{T2},
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine, T2<:RHFTEIEngine}
  comm = MPI.COMM_WORLD
  indicies = get_df_static_basis_indices(basis_sets, MPI.Comm_size(comm), MPI.Comm_rank(comm))
  if iteration == 1
    two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
    three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options)
    calculate_D_BLAS_NoThread!(scf_data, two_center_integrals, three_center_integrals, basis_sets, indicies, scf_options)
  end  
  calculate_coulomb_BLAS_NoThread!(scf_data, occupied_orbital_coefficients , basis_sets, indicies,scf_options)
  calculate_exchange_BLAS_NoThread!(scf_data, occupied_orbital_coefficients ,basis_sets, indicies,scf_options)

end

function calculate_D_BLAS_NoThread_sym!(scf_data, two_center_integrals, three_center_integrals, basis_sets, indicies, scf_options::SCFOptions)
  μμ = scf_data.μ
  νν = scf_data.μ
  AA = length(indicies)
  # triangle_size = μμ*(μμ+1)÷2
  # three_eri_triangle = zeros(Float64, (triangle_size, AA))
  # Threads.@threads for A in 1:AA
  #   three_eri_triangle[:,A] = vec_triu_loop(view(three_center_integrals, :,:,A))
  # end

  J_AB_INV = convert(Array, inv(cholesky(Hermitian(two_center_integrals, :L)).U))
  if MPI.Comm_size(MPI.COMM_WORLD) > 1
    J_AB_INV = J_AB_INV[:,indicies]
  end
  BLAS.gemm!('N', 'N', 1.0, reshape(three_center_integrals, (μμ*νν,scf_data.A)), J_AB_INV, 0.0, reshape(scf_data.D, (μμ*νν,AA)))
  BLAS.gemm!('N', 'N', 1.0, three_eri_triangle, J_AB_INV, 0.0, scf_data.D_triangle)
 

end


function calculate_coulomb_BLAS_NoThread_sym!(scf_data, occupied_orbital_coefficients,  basis_sets, indicies, scf_options :: SCFOptions)
  AA = length(indicies)
  # BLAS.symm!('L', 'L', 1.0, occupied_orbital_coefficients, occupied_orbital_coefficients, 0.0, scf_data.density)


  BLAS.gemm!('N', 'T', 1.0, occupied_orbital_coefficients, occupied_orbital_coefficients, 0.0, scf_data.density)
  BLAS.gemm!('N', 'N', 1.0,  reshape(scf_data.density, (1,scf_data.μ*scf_data.μ)), reshape(scf_data.D, (scf_data.μ*scf_data.μ,AA)), 
  0.0, reshape(scf_data.coulomb_intermediate, (1,AA)))


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

  # den = vec_triu_loop_double(scf_data.density)
  # DD = zeros(Float64, (length(den), AA))
  # scf_data.D[1,2,1]
  # for A in 1:AA
  #   DD[:,A] = vec_triu_loop(view(scf_data.D, :,:,A))
  # end

  # BLAS_number_of_threads = BLAS.get_num_threads()
  # BLAS.set_num_threads(1)
  # Threads.@threads for A in 1:AA
  #   BLAS.gemm!('T', 'N', 1.0,  den, view(DD, :,A), 0.0, view(scf_data.coulomb_intermediate, A:A))
  # end

  # BLAS.gemm!('T', 'N', 1.0,  den, scf_data.D_triangle, 0.0, reshape(scf_data.coulomb_intermediate, (1,AA)))


  # BLAS.set_num_threads(BLAS_number_of_threads)
  # reshape(scf_data.D, (scf_data.μ*scf_data.μ, AA))


  # BLAS.gemm!('N', 'N', 2.0, scf_data.D_triangle, scf_data.coulomb_intermediate, 0.0, scf_data.two_electron_fock_triangle)

  # ii = 1
  # for i in 1:scf_data.μ
  #   for j in 1:i-1
  #     scf_data.two_electron_fock[i,j] = scf_data.two_electron_fock_triangle[ii]
  #     scf_data.two_electron_fock[j,i] = scf_data.two_electron_fock_triangle[ii]
  #     ii+=1
  #   end
  #   scf_data.two_electron_fock[i,i] = scf_data.two_electron_fock_triangle[ii]
  #   ii+=1
  # end

end

function calculate_exchange_BLAS_NoThread_sym!(scf_data, occupied_orbital_coefficients,  basis_sets, indicies, scf_options :: SCFOptions)
  AA = length(indicies)
  μμ = scf_data.μ
  ii = scf_data.occ

  BLAS.gemm!('T', 'N' , 1.0, reshape(scf_data.D, (μμ, μμ*AA)), occupied_orbital_coefficients, 0.0, reshape(scf_data.D_tilde, (μμ*AA,ii)))


  # Threads.@threads for A in 1:AA
  #   for i in 1:ii
  #     for p in 1:μμ
  #       scf_data.D_tilde[i,A,p] = 0.0
  #       for r in 1:μμ
  #         scf_data.D_tilde[i,A,p] += occupied_orbital_coefficients[r,i]*scf_data.D[p,r,A]
  #       end
  #     end
  #   end
  # end 


  Threads.@threads for p in 1:μμ
    for q in 1:p
      BLAS.gemm!('N', 'N', -1.0, reshape(view(scf_data.D_tilde, :, :, p), (1,ii*AA,)), reshape(view(scf_data.D_tilde, :, :, q), (ii*AA, 1)), 1.0, view(scf_data.two_electron_fock, p:p,q:q))
      if p != q
        scf_data.two_electron_fock[q,p] =  scf_data.two_electron_fock[p,q]
      end
    end
  end

  # for i in 1:ii
  #   for A in 1:AA
  #     for p in 1:μμ
  #       println("scf_data.D_tilde[$i,$A,$p] = ", scf_data.D_tilde[i,A,p])
  #     end
  #   end
  # end

  # BLAS.gemm!('T', 'N' , 1.0, reshape(scf_data.D, (μμ, μμ*AA)), occupied_orbital_coefficients, 0.0, reshape(scf_data.D_tilde, (μμ*AA,ii)))
  # BLAS.gemm!('N', 'T', -1.0, reshape(scf_data.D_tilde, (μμ, ii*AA)), reshape(scf_data.D_tilde, (μμ, ii*AA)), 1.0, scf_data.two_electron_fock)
  
end

@inline function twoD_to1Dindex(i,j, p)
  return (i-1)*p + j
end 

# function serialize_data(file_name, data, iteration)

#   if iteration != 1
#     return 
#   end
#   #get directory of current running script

#   serialization_path = "$(pwd())/testoutputs/C20H42/screened_no_sym"

#   try
#     # run(`rm -rf $serialization_path`)
#   catch
#   end
#   try
#     mkpath(serialization_path)
#   catch
#   end
 
#   io = open("$serialization_path/$file_name", "w")
#   serialize(io, data)
#   close(io)
# end


function serialize_data(file_name, data, iteration)

  if iteration != 1
    return 
  end
  #get directory of current running script

  serialization_path = "$(pwd())/testoutputs/water/for_hackathon_hdf5"

  try
    # run(`rm -rf $serialization_path`)
  catch
  end
  try
    mkpath(serialization_path)
  catch
  end

  data_size = size(data)


 
  # io = open("$serialization_path/$file_name", "w+")

  # write(io, length(data_size))
  # #row major for c++ use
  # if length(data_size) == 3
  #   for i in 1:data_size[1]
  #     for j in 1:data_size[2]
  #       for k in 1:data_size[3]
  #       write(io, "$(data[i,j,k])\n")
  #       end 
  #     end
  #   end
  # else
  #   for i in 1:data_size[1]
  #     for j in 1:data_size[2]
  #       write(io, "$(data[i,j])\n")
  #     end
  #   end
  # end
  #close(io)
  h5open("$serialization_path/$file_name", "w") do fid
    g = create_group(fid, "data")
    dset = create_dataset(g, "values", eltype(data), data_size)
    write(dset,data)
  end

end

function get_screening_metadata!(scf_data, jeri_engine_thread, two_center_integrals, basis_sets)

  max_P_P = get_max_P_P(two_center_integrals)

  scf_data.screening_data.shell_screen_matrix, 
  scf_data.screening_data.basis_function_screen_matrix, 
  scf_data.screening_data.sparse_pq_index_map = 
    schwarz_screen_itegrals_df(scf_data, 10^-12, max_P_P, basis_sets, jeri_engine_thread)
  basis_function_screen_matrix = scf_data.screening_data.basis_function_screen_matrix

  scf_data.screening_data.sparse_p_start_indices = zeros(Int64, scf_data.μ)
  scf_data.screening_data.non_screened_p_indices_count = zeros(Int64, scf_data.μ)
  screened_triangular_indicies = zeros(Int64, (scf_data.μ,scf_data.μ))
  non_zero_coefficients = []
  p_start = 1
  triangular_indices_count = 0 #the number of indicies that are unscreened and make up the lower triangle

  non_screened_p_indices_count = scf_data.screening_data.non_screened_p_indices_count
  sparse_p_start_indices = scf_data.screening_data.sparse_p_start_indices
  non_zero_coefficients = scf_data.screening_data.non_zero_coefficients
  scf_data.screening_data.screened_triangular_indicies = zeros(Int64, (scf_data.μ,scf_data.μ))
  screened_triangular_indicies = scf_data.screening_data.screened_triangular_indicies

  for pp in 1:scf_data.μ
    non_screened_p_indices_count[pp] = sum(basis_function_screen_matrix[:,pp]) 
    sparse_p_start_indices[pp] = p_start
    p_start+= non_screened_p_indices_count[pp]
    push!(non_zero_coefficients, zeros(non_screened_p_indices_count[pp], scf_data.occ))

    for qq in 1:scf_data.μ
      if qq >= pp && basis_function_screen_matrix[qq,pp]
          triangular_indices_count += 1
          screened_triangular_indicies[qq,pp] = triangular_indices_count
          screened_triangular_indicies[pp,qq] = triangular_indices_count
      end
    end
  end
  scf_data.screening_data.triangular_indices_count = triangular_indices_count
end


function get_triangle_matrix_length(n) :: Int
  return n*(n+1)÷2
end

function df_rhf_fock_build_screened!(scf_data, jeri_engine_thread_df::Vector{T}, jeri_engine_thread ::Vector{T2},
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine, T2<:RHFTEIEngine}
  comm = MPI.COMM_WORLD
  indicies = get_df_static_basis_indices(basis_sets, MPI.Comm_size(comm), MPI.Comm_rank(comm))

  p = scf_data.μ
  q = scf_data.μ
  Q = scf_data.A
  occ = scf_data.occ
  
  occupied_orbital_coefficients = permutedims(occupied_orbital_coefficients, (2,1))

  if iteration == 1
    println("doing triangular screening DF-RHF")
    two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
    get_screening_metadata!(scf_data, jeri_engine_thread, two_center_integrals, basis_sets)


    load = scf_options.load
    scf_options.load = "screened" #todo remove
    
    scf_data.D = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options, scf_data)
    scf_data.D = permutedims(scf_data.D, (2,1)) #todo put this in correct order in calulation step
    scf_options.load = load #todo remove
    

    J_AB_invt = convert(Array, inv(cholesky(Hermitian(two_center_integrals, :L)).U))

    # if MPI.Comm_size(MPI.COMM_WORLD) > 1
    #   J_AB_INV = J_AB_INV[:,indicies]
    # end

    BLAS.trmm!('L', 'U', 'T', 'N', 1.0, J_AB_invt, scf_data.D)
    scf_data.D_tilde = zeros(Float64, (scf_data.A, scf_data.occ, scf_data.μ))
    scf_data.J =  zeros(Float64, size(scf_data.D)[2])
    scf_data.K = zeros(Float64, size(scf_data.two_electron_fock))
    scf_data.coulomb_intermediate = zeros(Float64, scf_data.A)
    scf_data.density_array = zeros(Float64, size(scf_data.D)[2])

  end  

  screened_triangular_indices = scf_data.screening_data.screened_triangular_indicies
  basis_function_screen_matrix = scf_data.screening_data.basis_function_screen_matrix

  #todo move the screening indexes to iteration 1 and save them in scf_data
  println("coulomb")
  @time begin 
    BLAS.gemm!('T', 'N', 1.0, occupied_orbital_coefficients, occupied_orbital_coefficients, 0.0, scf_data.density)

    Threads.@threads for index in CartesianIndices(scf_data.density)
      if !basis_function_screen_matrix[index[1], index[2]] 
        continue
      end
      if index[1] == index[2]
        scf_data.density_array[screened_triangular_indices[index[1], index[2]]] = scf_data.density[index]
      else
        scf_data.density_array[screened_triangular_indices[index[1], index[2]]] = scf_data.density[index]*2.0 # trianglar symmetry necessitates doubling the value that is contracted into the coulomb intermediate
      end
    end


    BLAS.gemv!('N', 1.0, scf_data.D, scf_data.density_array, 0.0, scf_data.coulomb_intermediate)
    BLAS.gemv!('T', 2.0, scf_data.D, scf_data.coulomb_intermediate , 0.0, scf_data.J)    
  
  
    scf_data.two_electron_fock .= 0.0
    Threads.@threads for index in CartesianIndices(scf_data.two_electron_fock)
      if !basis_function_screen_matrix[index] || index[2] > index[1]
        continue
      end
      scf_data.two_electron_fock[index] = scf_data.J[screened_triangular_indices[index]]
      scf_data.two_electron_fock[index[2], index[1]] = scf_data.two_electron_fock[index]
    end
  end
  #make the below code a bit more readable/shorter with a couple variable 
  println("exchange")
  B_buffer = zeros(Float64, (Q, p, Threads.nthreads()))
  @time begin 
    @time calculate_W_from_trianglular_screened(scf_data.D, B_buffer, p, occupied_orbital_coefficients, 
      basis_function_screen_matrix, 
      screened_triangular_indices, 
      scf_data.screening_data.non_zero_coefficients, scf_data.D_tilde)
    @time calculate_K_upper_diagonal_block(scf_data)  
  end   
end

function calculate_K_upper_diagonal_block(scf_data)
  W = scf_data.D_tilde
  K = scf_data.K
  p = scf_data.μ
  Q = scf_data.A
  occ = scf_data.occ
  basis_function_screen_matrix = scf_data.screening_data.basis_function_screen_matrix
  W_view = reshape(W, Q * occ, p)



  block_size = 0
  for i in 20:-1:10
      if p%i == 0
          block_size = p÷i
          break
      end
  end
  if block_size == 0 # deal with numbers where block size to fit directly without unsquare section 
      block_size = max(64, p÷10)
  end
  if p < block_size
      block_size = p÷2 #small matrix just use half of the matrix dimension. 
  end  

  number_of_non_screened_blocks = zeros(Int, p ÷ block_size, p ÷ block_size)

  p_views = Array{SubArray}(undef, p ÷ block_size)
  K_views = Array{SubArray}(undef, p ÷ block_size, p ÷ block_size)
  screened_blocks = zeros(Bool, p ÷ block_size, p ÷ block_size)

  #todo do this once in iteration 1 only
  Threads.@threads for qq in 1:p÷block_size
    p_views[qq] = view(W_view, :, block_size*(qq-1)+1:block_size*(qq-1)+1+block_size-1)
    for pp in 1:qq
      screened_blocks[qq, pp] = 0 == sum(
        view(basis_function_screen_matrix,
          block_size*(qq-1)+1:block_size*(qq-1)+1+block_size-1,
          block_size*(pp-1)+1:block_size*(pp-1)+1+block_size-1))
      screened_blocks[pp, qq] = screened_blocks[qq, pp]

      K_views[pp, qq] = view(K, block_size*(pp-1)+1:block_size*(pp-1)+1+block_size-1, block_size*(qq-1)+1:block_size*(qq-1)+1+block_size-1)
    end
  end


  blas_threads = BLAS.get_num_threads()
  BLAS.set_num_threads(1)
    Threads.@threads for qq in 1:p÷block_size # iterate over column blocks
      for pp in 1:qq # iterate over row blocks
        if screened_blocks[pp, qq]
          continue
        end
        number_of_non_screened_blocks[pp, qq] = 1
        BLAS.gemm!('T', 'N', -1.0,
          p_views[pp],
          p_views[qq],
          0.0,
          K_views[pp, qq])
      end
    end
  BLAS.set_num_threads(blas_threads)

  #non square part of K not include in the blocks above if any are non screened
    q_range = p-(p%block_size-1):p
    p_range = 1:p
    if length(q_range) > 0
      number_of_non_screened_pq_pairs = sum(basis_function_screen_matrix[q_range,p_range])
      if number_of_non_screened_pq_pairs != 0
        BLAS.gemm!('T', 'N', -1.0, view(W_view, :, p_range), view(W_view, :, q_range),
          0.0, view(K, p_range, q_range))
      end
    end

  Threads.@threads for pp in 1:p
    for qq in 1:pp-1
      K[pp, qq] = K[qq, pp]
    end
  end
  axpy!(1.0, K, scf_data.two_electron_fock )
end


function calculate_W_from_trianglular_screened(B, B_buffer, p,occupied_orbital_coefficients, basis_function_screen_matrix, screened_triangular_indices, non_zero_coefficients, W)
  # blas_threads = BLAS.get_num_threads()
  # BLAS.set_num_threads(1)
  #divide the pp indicies and process on different threads
  indicies_per_thread = p÷Threads.nthreads()
  Threads.@threads for thread in 1:Threads.nthreads()
    # iterate over pp indicies for the thread
    thread_pp_range = (thread-1)*indicies_per_thread+1:thread*indicies_per_thread
    println("thread $(thread) thread_pp_range = ", thread_pp_range)
    if thread == Threads.nthreads()
        thread_pp_range = (thread-1)*indicies_per_thread+1:p
    end
    for pp in thread_pp_range
      non_zero_r_index = 1
      for r in eachindex(view(screened_triangular_indices, :, pp))
          if basis_function_screen_matrix[r,pp] 
              non_zero_coefficients[pp][non_zero_r_index, :] .= occupied_orbital_coefficients[:,r] 
              non_zero_r_index += 1
          end
      end
      indices = [x for x in view(screened_triangular_indices, :, pp) if x != 0]
      B_buffer[:, 1:length(indices), thread] .= view(B, :, indices)
      BLAS.gemm!('N', 'N', 1.0,  B_buffer[:, 1:length(indices), thread], non_zero_coefficients[pp], 0.0, view(W, :, :, pp))
    end 
  end
  # BLAS.set_num_threads(blas_threads)
end



function df_rhf_fock_build_BLAS!(scf_data, jeri_engine_thread_df::Vector{T}, basis_sets::CalculationBasisSets,
    occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}
  comm = MPI.COMM_WORLD
  indicies = get_df_static_basis_indices(basis_sets, MPI.Comm_size(comm), MPI.Comm_rank(comm))
  if iteration == 1
    two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
    three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options)
    calculate_D_BLAS_NoThread!(scf_data, two_center_integrals, three_center_integrals, basis_sets, indicies, scf_options)
  end  
  @time calculate_coulomb_BLAS_NoThread!(scf_data, occupied_orbital_coefficients , basis_sets, indicies,scf_options)
  @time calculate_exchange_BLAS_NoThread!(scf_data, occupied_orbital_coefficients ,basis_sets, indicies,scf_options)
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
  BLAS.gemm!('N', 'T', 1.0, occupied_orbital_coefficients, occupied_orbital_coefficients, 0.0, scf_data.density)
  

  BLAS.gemv!('T', 1.0, reshape(scf_data.D, (scf_data.μ*scf_data.μ,AA)), reshape(scf_data.density, scf_data.μ*scf_data.μ), 0.0, scf_data.coulomb_intermediate)
  BLAS.gemv!('N', 2.0, reshape(scf_data.D, (scf_data.μ*scf_data.μ,AA)), scf_data.coulomb_intermediate , 0.0, reshape(scf_data.two_electron_fock, scf_data.μ^2))

  
  # BLAS.gemm!('N', 'N', 1.0,  reshape(scf_data.density, (1,scf_data.μ*scf_data.μ)), reshape(scf_data.D, (scf_data.μ*scf_data.μ,AA)), 
  #   0.0, reshape(scf_data.coulomb_intermediate, (1,AA)))
  # BLAS.gemm!('N', 'N', 2.0, reshape(scf_data.D, (scf_data.μ*scf_data.μ, AA)), scf_data.coulomb_intermediate, 0.0,
  #   reshape(scf_data.two_electron_fock,(scf_data.μ*scf_data.μ, 1)))
end

function calculate_exchange_BLAS_NoThread!(scf_data, occupied_orbital_coefficients,  basis_sets, indicies, scf_options :: SCFOptions)
  AA = length(indicies)
  μμ = scf_data.μ
  ii = scf_data.occ
  println("size(scf_data.D_tilde) = $(size(scf_data.D_tilde))")
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