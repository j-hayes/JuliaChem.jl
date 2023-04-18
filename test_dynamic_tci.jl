using MPI
using LinearAlgebra
using Random
function run_three_center_integrals_dynamic(cartesian_indices, three_center_integrals, basis_length)
    comm = MPI.COMM_WORLD
    
    three_center_integrals_workspace = deepcopy(three_center_integrals)
    nindices = length(cartesian_indices)
    batches_per_thread = basis_length
    batch_size = basis_length

    n_threads = Threads.nthreads()

    if MPI.Comm_rank(comm) == 0
      println("n_threads: $(n_threads)")
      println("nindices: $(nindices)")
      println("worker ranks : $(MPI.Comm_size(comm)-1)")
      worker_thread_count = n_threads*(MPI.Comm_size(comm) - 1)
      println("worker threads: $(worker_thread_count)")
      println("batch_size: $(batch_size)")
    end
    task_starts = []
    task_ends = []
    task_rank = []
    task_thread = []
   
    #== controller rank ==#
    if MPI.Comm_rank(comm) == 0 
      #== send out initial tasks to worker threads ==#
      task = [ nindices]
      initial_task = 1
  
      recv_mesg_master = [ 0 ]
     
      #println("Start sending out initial tasks") 
      while initial_task < MPI.Comm_size(comm)
        for thread in 1:Threads.nthreads()
          sreq = MPI.Isend(task, initial_task, thread, comm)       
          # println("task: $(task) sent to rank: $(initial_task) thread: $(thread)")
          append!(task_starts, task[1])
          append!(task_ends, task[1] - batch_size)
          append!(task_rank, initial_task)
          append!(task_thread, thread)
          task[1] -= batch_size + 1
        end
        initial_task += 1
      end

      #== hand out integrals to worker threads dynamically ==#
      while task[1] > 0 
        status = MPI.Probe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, 
          comm) 

        sreq = MPI.Isend(task, status.source, status.tag, comm) 
        # println("task: $(task) sent to rank: $(status.source) thread: $(status.tag)") 
        append!(task_starts, task[1])
        append!(task_ends, task[1] - batch_size)
        append!(task_rank, status.source)
        append!(task_thread, status.tag)

        task[1] -= batch_size + 1
      end     
      #== hand out ending signals once done ==#
      for rank in 1:(MPI.Comm_size(comm)-1)
        for thread in 1:Threads.nthreads()
          sreq = MPI.Isend([ -1 ], rank, thread, comm)                           
        end
      end      
      
    #== worker ranks perform actual computations on quartets ==#
    else
      #== create needed mutices ==#
      mutex_mpi_worker = Base.Threads.ReentrantLock()
      #== execute kernel ==# 
      @sync for thread in 1:Threads.nthreads()
        Threads.@spawn begin 
          #== initial set up ==#
          recv_mesg = [ 0 ] 
          send_mesg = [ MPI.Comm_rank(comm) ]   
          #== complete first task ==#
          lock(mutex_mpi_worker)
            status = MPI.Probe(0, thread, comm)
            rreq = MPI.Recv!(recv_mesg, status.source, status.tag, comm)
            ijkl_index = recv_mesg[1]
          unlock(mutex_mpi_worker)    

          println("rank $(MPI.Comm_rank(comm)), thread: $thread doing integrals: $(ijkl_index):$(ijkl_index-batch_size)\n")

          for ijkl in ijkl_index:-1:(max(1,ijkl_index-batch_size))
            # println("ijkl: $(ijkl)\n")
            index = cartesian_indices[ijkl]
            three_center_integrals_workspace[index] = 1.0 #100*MPI.Comm_rank(comm) + thread
          end
          #== complete rest of tasks ==#
          while ijkl_index >= 1 
            flush(stdout)
            lock(mutex_mpi_worker)
              status = MPI.Sendrecv!(send_mesg, 0, thread, recv_mesg, 0, thread, comm)
              ijkl_index = recv_mesg[1]
            unlock(mutex_mpi_worker)

            println("rank $(MPI.Comm_rank(comm)), thread: $thread doing integrals: $(ijkl_index):$(ijkl_index-batch_size)\n")
            for ijkl in ijkl_index:-1:(max(1,ijkl_index-batch_size))
              # println("ijkl: $(ijkl)\n")
              index = cartesian_indices[ijkl]
              if ijkl == 1
                println("doing index: $(ijkl) on rank: $(MPI.Comm_rank(comm)) thread: $thread")
              end
              three_center_integrals_workspace[index] = 1.0 #100*MPI.Comm_rank(comm) + thread
              
            end
          end
        end
      end     
    end

    MPI.Barrier(comm)
    three_center_integrals .= MPI.Allreduce(three_center_integrals_workspace, MPI.SUM, MPI.COMM_WORLD)
    MPI.Barrier(comm)



    number_of_missing_indices = 0
    if MPI.Comm_rank(comm) == 0
      for i in eachindex(task_starts)
        println("start:end [$i]: $(task_starts[i]):$(task_ends[i]) on rank: $(task_rank[i]) thread: $(task_thread[i])")
        if i > 1
          if task_starts[i] + 1 != task_ends[i-1] 
            println("ERROR: task_starts[$i] + 1 != task_ends[$(i-1)]")
          end
        end
      end
      
      for index in eachindex(three_center_integrals)
          println("TCI[,$index,]: $(three_center_integrals[index])")
          if three_center_integrals[index] == 0
              number_of_missing_indices += 1
          end
      end
      println("number_of_missing_indices: $(number_of_missing_indices)")


    end

    MPI.Barrier(comm)
end

function run_two_center_integrals_dynamic!(two_center_integrals)
  comm = MPI.COMM_WORLD
  n_threads = Threads.nthreads()
  n_integrals = length(two_center_integrals)
  batch_size = size(two_center_integrals, 1)
  rank = MPI.Comm_rank(comm)
  n_ranks = MPI.Comm_size(comm)
  task_top_index = n_integrals
  if rank == 0
    println("n_integrals: $(n_integrals)")
    println("batch_size: $(batch_size)")
    setup_two_center_integral_coordinator(task_top_index, batch_size, n_ranks, n_threads)
    flush(stdout)
  else
    run_two_center_integrals_worker(two_center_integrals, batch_size)
    flush(stdout)

  end

  MPI.Barrier(comm)
  two_center_integrals .= MPI.Allreduce(two_center_integrals, MPI.SUM, MPI.COMM_WORLD)
  MPI.Barrier(comm)

  missing_integrals = 0
  if rank == 0
    for index in eachindex(two_center_integrals)
      if two_center_integrals[index] == 0
        missing_integrals += 1
      end
      # println("TCI[,$index,]: $(two_center_integrals[index])")
    end
    println("missing_integrals: $(missing_integrals)")
  end

end

function setup_two_center_integral_coordinator(task_top_index, batch_size, n_ranks, n_threads)
  task_top_index = send_initial_tasks_two_center_integral_workers!(task_top_index, batch_size, n_ranks, n_threads)
  send_two_center_integral_tasks_dynamic(task_top_index, batch_size)
  send_end_signals(n_ranks, n_threads)
end

function send_end_signals(n_ranks, n_threads)
  for rank in 1:(n_ranks-1)
    for thread in 1:n_threads
      sreq = MPI.Isend([ -1 ], rank, thread, MPI.COMM_WORLD)                           
    end
  end    
end

function send_initial_tasks_two_center_integral_workers!(task_top_index, batch_size, n_ranks, n_threads)
  for rank in 1:n_ranks-1
    for thread in 1:n_threads
      sreq = MPI.Isend([ task_top_index ], rank, thread, MPI.COMM_WORLD)
      task_top_index -= batch_size + 1
    end
  end
  return task_top_index
end

function send_two_center_integral_tasks_dynamic(task_top_index, batch_size)
  while task_top_index > 0 
    status = MPI.Probe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, MPI.COMM_WORLD) 
    sreq = MPI.Isend([ task_top_index ], status.source, status.tag, MPI.COMM_WORLD) 
    println("task: $(task_top_index) sent to rank: $(status.source) thread: $(status.tag) \n") 
    task_top_index -= batch_size + 1
  end     
end

function run_two_center_integrals_worker(two_center_integrals, batch_size)
  comm = MPI.COMM_WORLD
  mutex_mpi_worker = Base.Threads.ReentrantLock()
  #== execute kernel ==# 
  @sync for thread in 1:Threads.nthreads()
    Threads.@spawn begin 
      #== initial set up ==#
      recv_mesg = [ 0 ] 
      send_mesg = [ MPI.Comm_rank(comm) ]   
      #== complete first task ==#
      lock(mutex_mpi_worker)
        status = MPI.Probe(0, thread, comm)
        rreq = MPI.Recv!(recv_mesg, status.source, status.tag, comm)
        ij_index = recv_mesg[1]
      unlock(mutex_mpi_worker)    
      # println("rank $(MPI.Comm_rank(comm)), thread: $thread doing integrals: $(ij_index):$(ij_index-batch_size)\n")

      do_two_center_integral_batch(two_center_integrals, ij_index, ij_index-batch_size)
      while ij_index >= 1 
        ij_index = get_next_batch(mutex_mpi_worker, send_mesg, recv_mesg, comm, thread)
        # println("doing next batch on rank: $(MPI.Comm_rank(comm)) thread: $thread with ij_index: $(ij_index)\n")
        do_two_center_integral_batch(two_center_integrals, ij_index, ij_index-batch_size)
      end
    end
  end
end

function get_next_batch(mutex_mpi_worker, send_mesg, recv_mesg, comm, thread)
  lock(mutex_mpi_worker)
    status = MPI.Sendrecv!(send_mesg, 0, thread, recv_mesg, 0, thread, comm)
    ij_index = recv_mesg[1]
  unlock(mutex_mpi_worker)
  return ij_index
end

function do_two_center_integral_batch(two_center_integrals, top_index, bottom_index)
  for ij in top_index:-1:(max(1,bottom_index))
    # println("ijkl: $(ijkl)\n")
    two_center_integrals[ij] = 1.0 #100*MPI.Comm_rank(comm) + thread
  end
end


function main()
  MPI.Init()
  if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println("n threads = $(Threads.nthreads())")
    println("n ranks = $(MPI.Comm_size(MPI.COMM_WORLD))")
    println("n worker ranks = $(MPI.Comm_size(MPI.COMM_WORLD) - 1)")
  end
  # three_center_integrals = zeros(13,13,96)
  # three_center_integrals = zeros(13,13,96)
  # cartesian_indices = CartesianIndices(three_center_integrals)
  # run_three_center_integrals_dynamic(cartesian_indices, three_center_integrals, size(three_center_integrals, 1))

  two_center_integrals = zeros(50,50)
  run_two_center_integrals_dynamic!(two_center_integrals)
  # if MPI.Comm_rank(MPI.COMM_WORLD) == 0
  #   println("size of three_center_integrals: $(length(three_center_integrals))")
  #   println("size of cartesian_indices: $(length(cartesian_indices))")
  #   println("sum of three_center_integrals: $(sum(three_center_integrals))")
  # end
  MPI.Finalize()
end

main()