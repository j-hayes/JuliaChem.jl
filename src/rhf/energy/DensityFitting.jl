using Base.Threads
using LinearAlgebra
using TensorOperations

#== Density Fitted Restricted Hartree-Fock, Fock build step ==#
#== 
Indecies for all tensor contractions 
  duplicated letters: (e.g.) dd dummy variables that will be summed Overlap
  A = Auxillary Basis orbital
  μ,ν = Primary Basis orbital
  i = occupied orbitals

==#
@inline function df_rhf_fock_build(jeri_engine_thread::Vector{T}, basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, xyK, iteration, load) where {T<:DFRHFTEIEngine}
    comm = MPI.COMM_WORLD
    aux_basis_function_count = basis_sets.auxillary.norb
    basis_function_count = basis_sets.primary.norb
    two_electron_fock = zeros(Float64, (basis_function_count, basis_function_count))

  if iteration == 1
    two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread, basis_sets, load)
    three_center_integrals = calculate_three_center_integrals(jeri_engine_thread, basis_sets, load)
    calculate_xyK(two_center_integrals, three_center_integrals, xyK, load)
  end
  xiK = calculate_xiK(xyK, occupied_orbital_coefficients, basis_function_count, aux_basis_function_count, load)
  #Coulomb
  calculate_coulomb!(two_electron_fock, xyK, xiK, occupied_orbital_coefficients, basis_function_count ,load) 
  #Exchange
  calculate_exchange!(two_electron_fock, xiK, basis_function_count,load)
  MPI.Barrier(comm)

  return two_electron_fock
end

#original tensor operation  @tensor two_electron_fock[μ, ν] -= xiK[μ, νν, AA] * xiK[ν, νν, AA]
function calculate_exchange!(two_electron_fock, xiK, basis_function_count, load)
  # two_electron_fock_copy = deepcopy(two_electron_fock)
  # @tensor exchange[μ, ν] := xiK[μ, νν, AA] * xiK[ν, νν, AA]

  μμ = size(xiK, 1) # number of basis functions 
  oo = size(xiK, 2) # number of occupied orbitals
  AA = size(xiK, 3) # number of aux basis functions

  # transpose transpose gemm transpose tensor contraction 
  μ_vector =  reshape(xiK, (μμ, oo*AA))
  ν_vector =  reshape(xiK, (μμ, oo*AA))
  exchange_blas_vec = BLAS.gemm('N', 'T', 1.0, μ_vector, ν_vector)
  exchange_blas = reshape(exchange_blas_vec, (μμ, μμ))
  axpy!(-1.0, exchange_blas, two_electron_fock)
end

# exchange kernel 
@inline function calculate_exchange_kernel(two_electron_fock, xiK, index)
  μ = index[1]
  ν = index[2]
  return dot(view(xiK, μ,:, :), view(xiK, ν,:, :))
end


#original tensor operation  
#@tensoropt (μμ => 10, νν => 10) two_electron_fock_component[μ, ν] = 2.0 * xyK[μ, ν, A] * xiK[μμ, νν, A] * occupied_orbital_coefficients[μμ, νν]
@inline function calculate_coulomb_kernel(xyK, xiK, intermediate, index)
  μ = index[1]
  ν = index[2]
  return dot(view(xyK, μ, ν, :) , intermediate)
  # μ = index[1]
  # ν = index[2]
  # two_electron_fock_component[μ, ν] = 0.0
  # for A in axes(xyK, 3)
  #   for μμ in axes(occupied_orbital_coefficients, 1)
  #     for νν in axes(occupied_orbital_coefficients, 2)
  #       two_electron_fock_component[μ, ν] += 2.0 * xyK[μ, ν, A] * xiK[μμ, νν, A] * occupied_orbital_coefficients[μμ, νν]
  #     end
  #   end
  # end
end

@inline function calculate_coulomb!(two_electron_fock, xyK, xiK, occupied_orbital_coefficients, basis_function_count ,load) 
  
  io = open("coulomb.txt", "a")
  write(io, "xyK\n")
  write(io, "$(vec(xyK))\n")

  write(io, "xiK\n")
  write(io, "$(vec(xiK))\n")
  write(io, "occupied_orbital_coefficients\n")

  write(io, "$(vec(occupied_orbital_coefficients))\n")
  @tensoropt (μμ => 10, νν => 10) two_electron_fock[μ, ν] = 2.0 * xyK[μ, ν, A] * xiK[μμ, νν, A] * occupied_orbital_coefficients[μμ, νν]
  write(io, "two_electron_fock\n")
  write(io, "$(vec(two_electron_fock))\n")
  close(io)
  # μμ = size(xiK, 1) # number of basis functions
  # νν = size(xiK, 2) # number of basis functions
  # AA = size(xiK, 3) # number of AUX basis functions

  # xiK_matrix = permutedims(reshape(xiK, (μμ*νν, AA)), (2,1))
  # intermediate_matrix = BLAS.gemm('N', 'N', 1.0, xiK_matrix, occupied_orbital_coefficients)


  # comm = MPI.COMM_WORLD 
  # cartesian_indecies = CartesianIndices(two_electron_fock)
  # n_threads = Threads.nthreads()
  # number_of_indecies = length(cartesian_indecies)
  # batch_size = ceil(Int, number_of_indecies / n_threads)
  # thread_coulombs = [zeros(Float64, size(two_electron_fock)) for i in 1:n_threads]

  # intermediate = zeros(size(xiK,3)) 
  # for A in axes(xiK, 3)
  #   intermediate[A] = dot(view(xiK,:, :, A), occupied_orbital_coefficients)
  # end

  # if load == "sequential"
  #   for index in cartesian_indecies
  #     two_electron_fock[index] = calculate_coulomb_kernel(xyK, xiK, intermediate, index)
  #   end
  # elseif load == "static"    
  #   Threads.@sync for batch_index in 1:batch_size+1:number_of_indecies
  #     Threads.@spawn begin
  #       thread_coulomb = thread_coulombs[Threads.threadid()]
  #       for view_index in batch_index:min(number_of_indecies, batch_index + batch_size)
  #         index = cartesian_indecies[view_index]
  #         thread_coulomb[index] = calculate_coulomb_kernel(xyK, xiK, intermediate, index)
  #       end
  #       axpy!(2.0, thread_coulomb, two_electron_fock) 
  #     end
  #   end
  # end
  # MPI.Barrier(comm)
end 

@inline function calculate_xiK_kernel!(xyK, occupied_orbital_coefficients,occ_coef_indexes, index)
  μ = index[1] 
  i = index[2]
  A = index[3]
  xiK_value = 0.0 
  for dd in occ_coef_indexes
    xiK_value += xyK[μ, dd, A] * occupied_orbital_coefficients[dd, i]
  end
  return xiK_value
end

# calculate xiK 
# original tensor operation
# @tensor xiK[μ, i, A] = xyK[μ, νν, A] * occupied_orbital_coefficients[νν, i]

@inline function calculate_xiK(xyK, occupied_orbital_coefficients,
  basis_function_count, aux_basis_function_count, load)
  # io = open("xiK.txt", "a")
  # write(io, "xyK\n")
  # write(io, "$(vec(xyK))\n")
  # write(io, "occupied_orbital_coefficients\n")
  # write(io, "$(vec(occupied_orbital_coefficients))\n")
  # xyK_copy = deepcopy(xyK)
  # occupied_orbital_coefficients_copy = deepcopy(occupied_orbital_coefficients)
  # if scf_options.contraction_library == Constants.Contraction.Tensor_Op
  # @tensor xiK_tenop[ν, i, A] := xyK_copy[μμ, ν, A] * occupied_orbital_coefficients_copy[μμ, i]
    # return xiK_tenop
  #end 
  μμ = size(xyK, 1)
  νν = size(xyK, 2)
  AA = size(xyK, 3)
  ii = size(occupied_orbital_coefficients, 2)

  xyK_permuted = permutedims(xyK, (1,3,2))
  xyK_matrix =  reshape(xyK_permuted, (νν*AA,μμ))
  
  xiK_blas_vec = BLAS.gemm('N', 'N' , 1.0, xyK_matrix, occupied_orbital_coefficients)

  xiK_blas = permutedims(reshape(xiK_blas_vec, (μμ,AA,ii)), (1,3,2))
  return xiK_blas
end


#original tensor operation was:
# @tensor xyK[μ, ν, A] = three_center_integrals[dd, μ, ν]*Linv_t[dd, A]
@inline function calculate_xyK_kernel!(xyK, three_center_integrals, Linv_t, index)
  μ = index[1]
  ν = index[2]
  A = index[3]
  xyK[index] = dot(view(three_center_integrals,:, μ, ν), view(Linv_t, :, A))
end

@inline function calculate_xyK(two_center_integrals, three_center_integrals, xyK, load)
  comm = MPI.COMM_WORLD
  # this needs to be mpi parallelized
  Linv_t = convert(Array, transpose(cholesky(Hermitian(two_center_integrals, :L)).L \I))
  
  # # if scf_options.contraction_library == Constants.Contraction.Tensor_Op
  # @tensor xyK[μ, ν, A] = three_center_integrals[dd, μ, ν]*Linv_t[dd, A]
  # #   return
  # # end
  println("doing xyK blas")
  μμ = size(three_center_integrals, 2)
  νν = size(three_center_integrals, 3)
  AA = size(three_center_integrals, 1)
  # use transpose transpose gemm transpose to perform tensor contraction 
  # Linv_T is already a 2D matrix so no need to reshape, and is in correct order
  tci_vector =  reshape(permutedims(three_center_integrals, (2,3,1)) , (μμ*νν,AA))
  xyK_vec = BLAS.gemm('N', 'N', 1.0, tci_vector, Linv_t)
  xyK .= reshape(xyK_vec, (μμ, νν, AA))
  
  
  # cartesian_indecies = CartesianIndices(xyK)
  # number_of_indecies = length(cartesian_indecies)
  # n_threads = Threads.nthreads()
  # batch_size = ceil(Int, number_of_indecies / n_threads)
  # if load == "sequential"
  #   for index in cartesian_indecies
  #     calculate_xyK_kernel!(xyK, three_center_integrals, Linv_t, index)
  #   end
  # elseif load == "static" || MPI.Comm_size(comm) == 1
  #   @sync for batch_index in 1:batch_size+1:number_of_indecies
  #     Threads.@spawn begin
  #       do_xyK_batch(xyK, three_center_integrals, Linv_t, cartesian_indecies, batch_index, batch_size, number_of_indecies)
  #     end  
  #   end
  # end # end if
  # MPI.Barrier(comm)
end # end function calculate_xyK

@inline function do_xyK_batch(xyK, three_center_integrals, Linv_t, cartesian_indecies, batch_index, batch_size, number_of_indecies)
  for view_index in batch_index:min(number_of_indecies, batch_index + batch_size)
    calculate_xyK_kernel!(xyK, three_center_integrals, Linv_t, cartesian_indecies[view_index])
  end 
end

@inline function calculate_three_center_integrals(jeri_engine_thread, basis_sets::CalculationBasisSets, load)
  comm = MPI.COMM_WORLD

  aux_basis_function_count = basis_sets.auxillary.norb
  basis_function_count = basis_sets.primary.norb
  three_center_integrals = zeros(Float64, (aux_basis_function_count, basis_function_count, basis_function_count))
  auxilliary_basis_shell_count = length(basis_sets.auxillary)
  basis_shell_count = length(basis_sets.primary)

  cartesian_indecies = CartesianIndices((auxilliary_basis_shell_count, basis_shell_count, basis_shell_count))
  number_of_indecies = length(cartesian_indecies)
  n_threads = Threads.nthreads()
  batch_size = ceil(Int, number_of_indecies / n_threads)

  max_primary_nbas = max_number_of_basis_functions(basis_sets.primary)
  max_aux_nbas = max_number_of_basis_functions(basis_sets.auxillary)
  thead_integral_buffer = [zeros(Float64, max_primary_nbas^2*max_aux_nbas) for thread in 1:n_threads]
  if load == "sequential"
    engine = jeri_engine_thread[1]
    integral_buffer = thead_integral_buffer[1]
    for cartesian_index in cartesian_indecies
      calculate_three_center_integrals_kernel!(three_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)
    end
  elseif load == "static" || MPI.Comm_size(comm) == 1  # todo fix this multithreading issue
    Threads.@sync for batch_index in 1:batch_size+1:number_of_indecies
      Threads.@spawn begin
        do_three_center_integral_batch(batch_index,
         batch_size, cartesian_indecies, 
         three_center_integrals, jeri_engine_thread, 
         basis_sets, thead_integral_buffer, number_of_indecies)
      end
    end
  else
    error("integral threading load type: $(load) not supported")
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
        three_center_integrals[i, j, k] = values[values_index]
        values_index += 1
      end
    end
  end
end

@inline function calculate_two_center_intgrals(jeri_engine_thread::Vector{T}, basis_sets, load) where {T<:DFRHFTEIEngine}

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
  if load == "sequential"
    engine = jeri_engine_thread[1]
    for cartesian_index in cartesian_indicies
      integral_buffer = thead_integral_buffer[1]
      calculate_two_center_intgrals_kernel!(two_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)
    end
  elseif load == "static" || MPI.Comm_size(comm) == 1
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
    error("integral threading load type: $(load) not supported")
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

