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
  occupied_orbital_coefficients, xyK, xiK, two_electron_fock_component,
  iteration, load) where T <: DFRHFTEIEngine

  comm = MPI.COMM_WORLD
  aux_basis_function_count = basis_sets.auxillary.norb
  basis_function_count = basis_sets.primary.norb
  if iteration == 1
    three_center_integrals = Array{Float64}(undef, (aux_basis_function_count,basis_function_count,basis_function_count))
    two_center_integrals = zeros((aux_basis_function_count, aux_basis_function_count))
    calculate_three_center_integrals(jeri_engine_thread, basis_sets, three_center_integrals, load)
    calculate_two_center_intgrals(jeri_engine_thread, basis_sets, two_center_integrals, load)

    if MPI.Comm_rank(comm) == 0 
      hermitian_eri_block_2_center_matrix = Hermitian(two_center_integrals, :L)
      LLT_2_center = cholesky(hermitian_eri_block_2_center_matrix)
      two_center_cholesky_lower = LLT_2_center.L
      Linv_t = convert(Array, transpose(two_center_cholesky_lower \I))
      @tensor xyK[μ, ν, A] = three_center_integrals[dd, μ, ν]*Linv_t[dd, A]
    end
  end
  if MPI.Comm_rank(comm) == 0 
    @tensor   xiK[μ,i,A] = xyK[μ,dd,A]*occupied_orbital_coefficients[dd,i]
    #Coulomb

    @tensoropt (μμ=>10,νν=>10) two_electron_fock_component[μ, ν] = 2.0*xyK[μ, ν, A]*xiK[μμ, νν, A]*occupied_orbital_coefficients[μμ, νν]
    #Exchange

    @tensor two_electron_fock_component[μ, ν] -=  xiK[μ,νν,AA]*xiK[ν,νν,AA]
  end
  MPI.Barrier(comm)
  return two_electron_fock_component
end


@inline function calculate_three_center_integrals(jeri_engine_thread::Vector{T}, basis_sets::CalculationBasisSets, three_center_integrals, load) where T <: DFRHFTEIEngine
  comm = MPI.COMM_WORLD

  auxilliary_basis_shell_count = length(basis_sets.auxillary)
  basis_shell_count = length(basis_sets.primary)
  
  cartesian_indecies = eachindex(view(three_center_integrals, 1:auxilliary_basis_shell_count, 1:basis_shell_count, 1:basis_shell_count))
  number_of_indecies = length(cartesian_indecies)   
  n_threads = Threads.nthreads()
  batch_size = ceil(Int,number_of_indecies/n_threads)

  max_nbas = max(max_number_of_basis_functions(basis_sets.primary), max_number_of_basis_functions(basis_sets.auxillary))
  thead_integral_buffer = [Vector{Float64}(undef, max_nbas^3) for i in 1:n_threads]
  if load == "sequential"
    engine =  jeri_engine_thread[1]   
    integral_buffer =thead_integral_buffer[1]
    for cartesian_index in cartesian_indecies
      calculate_three_center_integrals_kernel!(three_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)   
    end
  elseif load  == "static"  || MPI.Comm_size(comm) == 1   
    @sync for batch_index in 1:batch_size:number_of_indecies
      Threads.@spawn begin
        thread_index = Threads.threadid()        
        engine =  jeri_engine_thread[thread_index]    
        integral_buffer = thead_integral_buffer[thread_index]                                     
        for view_index in batch_index:min(number_of_indecies, batch_index+batch_size)
          calculate_three_center_integrals_kernel!(three_center_integrals, engine, cartesian_indecies[view_index], basis_sets, integral_buffer)        
        end 
      end 
    end 
  elseif load == "dynamic" &&  MPI.Comm_size(comm) > 1   
    run_three_center_integrals_dynamic(cartesian_indecies, three_center_integrals, jeri_engine_thread, basis_sets, thead_integral_buffer)  
  else
    error("integral threading load type: $(load) not supported")
  end

end

@inline function run_three_center_integrals_dynamic(cartesian_indecies, three_center_integrals, jeri_engine_thread, basis_sets, thead_integral_buffer)  

  comm = MPI.COMM_WORLD
  n_threads = Threads.nthreads()
  comm = MPI.COMM_WORLD
  batches_per_thread = 1
  batch_size = 4
  three_center_integrals_thread = [ zeros(size(three_center_integrals)) for thread in 1:n_threads ]
  if MPI.Comm_rank(comm) == 0 
      task_index = [length(cartesian_indecies)]
      comm_index = 1
      while comm_index < MPI.Comm_size(comm)
          for thread in 1:n_threads
              sreq = MPI.Isend(task_index, comm_index, thread, comm)
              task_index[1] -= batch_size
          end         
          comm_index += 1
      end

      while task_index[1] > 0 
          status = MPI.Probe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, 
            comm) 
          #rreq = MPI.Recv!(recv_mesg_master, status.source, status.tag, 
          #  comm)  
          #println("Sending task $task to rank ", status.source)
          sreq = MPI.Isend(task_index, status.source, status.tag, comm)  
          #println("Task $task sent to rank ", status.source)
          task_index[1] -= batch_size 
        end

       for rank in 1:(MPI.Comm_size(comm)-1)
          for thread in 1:Threads.nthreads()
            sreq = MPI.Isend([ -1 ], rank, thread, comm)                           
          end
        end     
  elseif MPI.Comm_rank(comm) > 0
      mutex_mpi_worker = Base.Threads.ReentrantLock()
      @sync for thread_index in 1:n_threads
          Threads.@spawn begin 
              engine =  jeri_engine_thread[thread_index]    
              integral_buffer = thead_integral_buffer[thread_index]

              recv_mesg = [ 0 ] 
              send_mesg = [ MPI.Comm_rank(comm) ] 
              lock(mutex_mpi_worker)
              status = MPI.Sendrecv!(send_mesg, 0, $thread_index, recv_mesg, 0, 
              $thread_index, comm)
              top_index = recv_mesg[1]

              unlock(mutex_mpi_worker)

              for i in top_index:-1:(max(1,top_index-batch_size+1))
                calculate_three_center_integrals_kernel!(three_center_integrals_thread[thread_index], engine, cartesian_indecies[i], basis_sets, integral_buffer)    
              end
              
              
              #== complete rest of tasks ==#
              while top_index >= 1 
                  lock(mutex_mpi_worker)
                  status = MPI.Sendrecv!(send_mesg, 0, $thread_index, recv_mesg, 0, 
                      $thread_index, comm)
                  top_index = recv_mesg[1]
                  unlock(mutex_mpi_worker)

                  for i in top_index:-1:(max(1,top_index-batch_size+1))
                    calculate_three_center_integrals_kernel!(three_center_integrals_thread[thread_index], engine, cartesian_indecies[i], basis_sets, integral_buffer)    
                  end
              end

          end
      end

    for integrals in three_center_integrals_thread
      axpy!(1.0, integrals, three_center_integrals)
    end
  end
  MPI.Barrier(comm)
end

@inline function calculate_three_center_integrals_kernel!(three_center_integrals, engine, cartesian_index, basis_sets, integral_buffer:: Vector{Float64}) 
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

  n123 = n12 * shell_3_nbasis
  JERI.compute_eri_block_df(engine, integral_buffer, s1, s2, s3, n123, 0)      

  copy_values_to_output!(three_center_integrals, integral_buffer, bf_1_pos, bf_2_pos, bf_3_pos, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis)
  axial_normalization_factor(three_center_integrals, shell_1, shell_2, shell_3, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis, bf_1_pos, bf_2_pos, bf_3_pos)
end

@inline function copy_values_to_output!(three_center_integrals, values, bf_1_pos, bf_2_pos, bf_3_pos, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis)        
  temp_index = 1
  for i in bf_1_pos:bf_1_pos+shell_1_nbasis-1
    for j in bf_2_pos:bf_2_pos+shell_2_nbasis-1
      for k in bf_3_pos:bf_3_pos+shell_3_nbasis-1
        three_center_integrals[i,j,k] = values[temp_index]
        temp_index += 1
      end
    end
  end
end
@inline function calculate_two_center_intgrals(jeri_engine_thread::Vector{T}, basis_sets, two_center_integrals, load)  where T <: DFRHFTEIEngine
  auxilliary_basis_shell_count = length(basis_sets.auxillary)
  cartesian_indicies = eachindex(view(two_center_integrals, 1:auxilliary_basis_shell_count,1:auxilliary_basis_shell_count))
  number_of_indecies = length(cartesian_indicies)   
  n_threads = Threads.nthreads()
  batch_size = ceil(Int,number_of_indecies/n_threads)  


  max_nbas =  max_number_of_basis_functions(basis_sets.auxillary)
  thead_integral_buffer = [Vector{Float64}(undef, max_nbas^2) for i in 1:n_threads]
  aux_basis_function_count = basis_sets.auxillary.norb
  two_center_integrals_thread = [zeros((aux_basis_function_count, aux_basis_function_count)) for i in 1:n_threads]
  if load == "sequential"
    engine = jeri_engine_thread[1]
    for cartesian_index in cartesian_indicies
      integral_buffer = thead_integral_buffer[1]
      calculate_two_center_intgrals_kernel!(two_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)
    end  
  elseif load  == "static"  || load == "dynamic"
    comm = MPI.COMM_WORLD
    if MPI.Comm_rank(comm) == 0 
      @sync for batch_index in 1:batch_size:number_of_indecies
        Threads.@spawn begin
          thread_index = Threads.threadid()               
          engine =  jeri_engine_thread[thread_index]    
          integral_buffer = thead_integral_buffer[thread_index]
          for view_index in batch_index:min(number_of_indecies, batch_index+batch_size)
            calculate_two_center_intgrals_kernel!(two_center_integrals_thread[thread_index], engine, cartesian_indicies[view_index], basis_sets, integral_buffer)
          end 
        end 
      end 
      for integrals in two_center_integrals_thread
        axpy!(1.0, integrals, two_center_integrals)
      end   
    end
    
    MPI.Barrier(comm)    
  else
    error("integral threading load type: $(load) not supported")
  end
println("returning two center integrals")
end


@inline function calculate_two_center_intgrals_kernel!(two_center_integrals, engine, cartesian_index, basis_sets, integral_buffer:: Vector{Float64}) 
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

  JERI.compute_two_center_eri_block(engine, integral_buffer, shell_1_index-1, shell_2_index-1, shell_1_basis_count, shell_2_basis_count)     
  copy_values_to_output!(two_center_integrals, integral_buffer, shell_1, shell_2, shell_1_basis_count, shell_2_basis_count)
  axial_normalization_factor(two_center_integrals, shell_1, shell_2, shell_1_basis_count, shell_2_basis_count, bf_1_pos, bf_2_pos)
end


@inline function copy_values_to_output!(two_center_integrals, values, shell_1, shell_2, shell_1_basis_count, shell_2_basis_count)
  temp_index = 1
  for i in  shell_1.pos:shell_1.pos+shell_1_basis_count-1
      for j in shell_2.pos:shell_2.pos+shell_2_basis_count-1  
        if i >= j # makes sure we don't put any values onto the top triangle which happens for some reason sometimes 
          two_center_integrals[i,j] = values[temp_index]
        else        
          two_center_integrals[i,j] = 0.0
        end
        temp_index += 1       
      end 
  end
end

