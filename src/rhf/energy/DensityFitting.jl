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

@inline function df_rhf_fock_build!(two_electron_fock, jeri_engine_thread::Vector{T}, 
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, D, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}
  comm = MPI.COMM_WORLD

  if iteration == 1
  two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread, basis_sets, scf_options)
  three_center_integrals = calculate_three_center_integrals(jeri_engine_thread, basis_sets, scf_options)
  @time calculate_D(two_center_integrals, three_center_integrals, D, scf_options)
  end
  @time D_tilde = calculate_D_tilde(D, occupied_orbital_coefficients, scf_options)
  scf_data = SCFData(D, D_tilde, zeros(size(D_tilde)), two_electron_fock, 1,1,1)

  #Coulomb
  @time calculate_coulomb!(two_electron_fock,D, D_tilde,occupied_orbital_coefficients,scf_options, scf_data) 
  #Exchange
  @time calculate_exchange!(two_electron_fock, D_tilde,scf_options, scf_data)
  MPI.Barrier(comm)
  return two_electron_fock
end

#original tensor operation  @tensor two_electron_fock[μ, ν] -= xiK[μ, νν, AA] * xiK[ν, νν, AA]
function calculate_exchange!(two_electron_fock, D_tilde, scf_options::SCFOptions, scf_data)
  if scf_options.contraction_mode == ContractionMode.tensor_operations
    @tensor two_electron_fock[μ, ν] -= D_tilde[μ, νν, AA] * D_tilde[ν, νν, AA]
    return 
  end

  # todo get these from constants to make this more readable 
  μμ = size(D_tilde, 1) # number of basis functions 
  oo = size(D_tilde, 2) # number of occupied orbitals
  AA = size(D_tilde, 3) # number of aux basis functions

  # transpose transpose gemm transpose tensor contraction 
  μ_vector =  reshape(D_tilde, (μμ, oo*AA))
  ν_vector =  reshape(D_tilde, (μμ, oo*AA))
  exchange_blas_vec = BLAS.gemm('N', 'T', 1.0, μ_vector, ν_vector)
  exchange_blas = reshape(exchange_blas_vec, (μμ, μμ))
  BLAS.axpy!(-1.0, exchange_blas, two_electron_fock)
end

@inline function calculate_coulomb!(two_electron_fock, D, D_tilde, occupied_orbital_coefficients,
   scf_options::SCFOptions, scf_data) 
     
  scf_data.D_tilde_permuted = permutedims(D_tilde, (2,1,3))
  if scf_options.contraction_mode == ContractionMode.tensor_operations 
    @tensoropt (μμ => 10, ii => 10) two_electron_fock[μ, ν] = 2.0 * D[μ, ν, A] * 
    scf_data.D_tilde_permuted[A,μμ,ii] * occupied_orbital_coefficients[μμ, ii]
    return 
  end

  μμ = size(D, 1)
  ii = size(occupied_orbital_coefficients, 2)
  AA = size(D, 3)

  scf_data.D_tilde_permuted = reshape(scf_data.D_tilde_permuted, (AA,μμ*ii))

  occupied_orbital_coefficients_vector = reshape(occupied_orbital_coefficients, (μμ*ii,1))
  intermediate_blas = BLAS.gemm('N', 'N', 1.0, scf_data.D_tilde_permuted, occupied_orbital_coefficients_vector)
  intermediate_blas = reshape(intermediate_blas, (AA,1))

  μμ = size(D, 1)
  νν = size(D, 2)
  AA = size(D, 3)

  xyK_matrix = reshape(D, (μμ*νν, AA))
  two_electron_fock = reshape(two_electron_fock, (μμ*νν, 1))
  BLAS.gemm!('N', 'N', 2.0, xyK_matrix, intermediate_blas, 0.0 ,two_electron_fock)
  two_electron_fock = reshape(two_electron_fock, (μμ, νν)) 

end 

# calculate xiK 
@inline function calculate_D_tilde(D, occupied_orbital_coefficients, scf_options::SCFOptions)
  if scf_options.contraction_mode == ContractionMode.tensor_operations
    @tensor D_tilde[ν, A, i] := D[μμ, ν, A] * occupied_orbital_coefficients[μμ, i]
    return D_tilde
  end 
  μμ = size(D, 1)
  νν = size(D, 2)
  AA = size(D, 3)
  ii = size(occupied_orbital_coefficients, 2)

  D_permuted = permutedims(D, (1,3,2))
  D_permuted = reshape(D_permuted, (νν*AA,μμ))
  d_tilde_vec = BLAS.gemm('N', 'N' , 1.0, D_permuted, occupied_orbital_coefficients)

  d_tilde = reshape(d_tilde_vec, (μμ,AA,ii))

  return d_tilde
end


@inline function calculate_D(two_center_integrals, three_center_integrals, D, scf_options::SCFOptions)
  comm = MPI.COMM_WORLD
  # this needs to be mpi parallelized
  J_AB_invt = convert(Array, transpose(cholesky(Hermitian(two_center_integrals, :L)).L \I))
  
  if scf_options.contraction_mode == ContractionMode.tensor_operations
    @tensor D[μ, ν, A] = three_center_integrals[μ, ν, BB]*J_AB_invt[BB, A]
    return
  end
  
  ## todo get this from a parameter to make this more readable 
  μμ = size(three_center_integrals, 1)
  νν = size(three_center_integrals, 2)
  AA = size(three_center_integrals, 3)
  # use transpose transpose gemm transpose to perform tensor contraction 
  # Linv_T is already a 2D matrix so no need to reshape, and is in correct order
  D = reshape(D, (μμ*νν,AA))
  three_center_integrals =  reshape(three_center_integrals, (μμ*νν,AA))
  BLAS.gemm!('N', 'N', 1.0, three_center_integrals, J_AB_invt, 0.0, D)
  D = reshape(D, (μμ, νν, AA))
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
  elseif scf_options.load == "static" || MPI.Comm_size(comm) == 1  # todo fix this multithreading issue
    Threads.@sync for batch_index in 1:batch_size+1:number_of_indecies
      Threads.@spawn begin
        do_three_center_integral_batch(batch_index,
         batch_size, cartesian_indecies, 
         three_center_integrals, jeri_engine_thread, 
         basis_sets, thead_integral_buffer, number_of_indecies)
      end
    end
  else
    error("integral threading load type: $(scf_options.load) not supported")
  end
  return three_center_integrals
  MPI.Barrier(comm)
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

# @inline function run_three_center_integrals_dynamic(cartesian_indecies, three_center_integrals, jeri_engine_thread, basis_sets, thead_integral_buffer)

#   comm = MPI.COMM_WORLD
#   n_threads = Threads.nthreads()
#   comm = MPI.COMM_WORLD
#   batches_per_thread = 1
#   batch_size = 4
#   three_center_integrals_thread = [zeros(size(three_center_integrals)) for thread in 1:n_threads]
#   if MPI.Comm_rank(comm) == 0
#     task_index = [length(cartesian_indecies)]
#     comm_index = 1
#     while comm_index < MPI.Comm_size(comm)
#       for thread in 1:n_threads
#         sreq = MPI.Isend(task_index, comm_index, thread, comm)
#         task_index[1] -= batch_size
#       end
#       comm_index += 1
#     end

#     while task_index[1] > 0
#       status = MPI.Probe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG,
#         comm)
#       #rreq = MPI.Recv!(recv_mesg_master, status.source, status.tag, 
#       #  comm)  
#       #println("Sending task $task to rank ", status.source)
#       sreq = MPI.Isend(task_index, status.source, status.tag, comm)
#       #println("Task $task sent to rank ", status.source)
#       task_index[1] -= batch_size
#     end

#     for rank in 1:(MPI.Comm_size(comm)-1)
#       for thread in 1:Threads.nthreads()
#         sreq = MPI.Isend([-1], rank, thread, comm)
#       end
#     end
#   elseif MPI.Comm_rank(comm) > 0
#     mutex_mpi_worker = Base.Threads.ReentrantLock()
#     @sync for thread_index in 1:n_threads
#       Threads.@spawn begin
#         engine = jeri_engine_thread[thread_index]
#         integral_buffer = thead_integral_buffer[thread_index]

#         recv_mesg = [0]
#         send_mesg = [MPI.Comm_rank(comm)]
#         lock(mutex_mpi_worker)
#         status = MPI.Sendrecv!(send_mesg, 0, $thread_index, recv_mesg, 0,
#           $thread_index, comm)
#         top_index = recv_mesg[1]

#         unlock(mutex_mpi_worker)

#         for i in top_index:-1:(max(1, top_index - batch_size + 1))
#           calculate_three_center_integrals_kernel!(three_center_integrals_thread[thread_index], engine, cartesian_indecies[i], basis_sets, integral_buffer)
#         end


#         #== complete rest of tasks ==#
#         while top_index >= 1
#           lock(mutex_mpi_worker)
#           status = MPI.Sendrecv!(send_mesg, 0, $thread_index, recv_mesg, 0,
#             $thread_index, comm)
#           top_index = recv_mesg[1]
#           unlock(mutex_mpi_worker)

#           for i in top_index:-1:(max(1, top_index - batch_size + 1))
#             calculate_three_center_integrals_kernel!(three_center_integrals_thread[thread_index], engine, cartesian_indecies[i], basis_sets, integral_buffer)
#           end
#         end

#       end
#     end

#     for integrals in three_center_integrals_thread
#       axpy!(1.0, integrals, three_center_integrals)
#     end
#   end
#   MPI.Barrier(comm)
# end

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

@inline function calculate_two_center_intgrals(jeri_engine_thread::Vector{T}, basis_sets, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}

  aux_basis_function_count = basis_sets.auxillary.norb
  two_center_integrals = zeros(Float64, aux_basis_function_count, aux_basis_function_count) 
  comm = MPI.COMM_WORLD
  auxilliary_basis_shell_count = length(basis_sets.auxillary)
  cartesian_indicies = CartesianIndices((auxilliary_basis_shell_count, auxilliary_basis_shell_count))
  number_of_indecies = length(cartesian_indicies)
  n_threads = Threads.nthreads()
  batch_size = ceil(Int, number_of_indecies / n_threads)

  max_nbas = max_number_of_basis_functions(basis_sets.auxillary)
  thead_integral_buffer = [zeros(Float64, max_nbas^2) for i in 1:n_threads]
  if scf_options.load == "sequential"
    engine = jeri_engine_thread[1]
    for cartesian_index in cartesian_indicies
      integral_buffer = thead_integral_buffer[1]
      calculate_two_center_intgrals_kernel!(two_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)
    end
  elseif scf_options.load == "static" || MPI.Comm_size(comm) == 1
    @sync for batch_index in 1:batch_size+1:number_of_indecies
      Threads.@spawn begin
        thread_id = Threads.threadid()
        for view_index in batch_index:min(number_of_indecies, batch_index + batch_size)
          cartesian_index = cartesian_indicies[view_index]
          engine = jeri_engine_thread[thread_id]
          integral_buffer = thead_integral_buffer[thread_id]
          calculate_two_center_intgrals_kernel!(two_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)
        end
      end
    end
  else
    error("integral threading load type: $(scf_options.load) not supported")
  end
  MPI.Barrier(comm)
  return two_center_integrals

end


@inline function calculate_two_center_intgrals_kernel!(two_center_integrals,
   engine, 
   cartesian_index, 
   basis_sets, 
   integral_buffer)
  shell_1_index = cartesian_index[1]
  shell_2_index = cartesian_index[2]

  if shell_2_index > shell_1_index #the top triangle of the symmetric matrix does not need to be calculated
    return
  end

  shell_1 = basis_sets.auxillary.shells[shell_1_index]
  shell_1_basis_count = shell_1.nbas
  bf_1_pos = shell_1.pos

  shell_2 = basis_sets.auxillary.shells[shell_2_index]
  shell_2_basis_count = shell_2.nbas
  bf_2_pos = shell_2.pos

  JERI.compute_two_center_eri_block(engine, integral_buffer, shell_1_index - 1, shell_2_index - 1, shell_1_basis_count, shell_2_basis_count)
  copy_values_to_output!(two_center_integrals, integral_buffer, shell_1, shell_2, shell_1_basis_count, shell_2_basis_count)
  axial_normalization_factor(two_center_integrals, shell_1, shell_2, shell_1_basis_count, shell_2_basis_count, bf_1_pos, bf_2_pos)
end


@inline function copy_values_to_output!(two_center_integrals, values, shell_1, shell_2, shell_1_basis_count, shell_2_basis_count)
  temp_index = 1
  for i in shell_1.pos:shell_1.pos+shell_1_basis_count-1
    for j in shell_2.pos:shell_2.pos+shell_2_basis_count-1
      if i >= j # makes sure we don't put any values onto the top triangle which happens for some reason sometimes 
        two_center_integrals[i, j] = values[temp_index]
      else
        two_center_integrals[i, j] = 0.0
      end
      temp_index += 1
    end
  end
end

