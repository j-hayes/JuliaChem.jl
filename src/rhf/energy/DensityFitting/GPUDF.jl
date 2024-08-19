using MPI
using CUDA
using CUDA.CUBLAS
using CUDA.CUSOLVER
using LinearAlgebra
using Base.Threads
using HDF5

function df_rhf_fock_build_GPU!(scf_data, jeri_engine_thread_df::Vector{T}, jeri_engine_thread::Vector{T2},
    basis_sets::CalculationBasisSets,
    occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine,T2<:RHFTEIEngine}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)
    pq = scf_data.μ^2

    p = scf_data.μ
    n_ooc = scf_data.occ

    devices = CUDA.devices()
    num_devices = length(devices)
    num_devices_global = num_devices*comm_size  
    scf_data.gpu_data.number_of_devices_used = num_devices
    occupied_orbital_coefficients = permutedims(occupied_orbital_coefficients, (2,1))

    three_center_integrals = Array{Array{Float64}}(undef, num_devices)
    if iteration == 1
        two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
        get_screening_metadata!(scf_data,
         scf_options.df_screening_sigma, jeri_engine_thread, two_center_integrals, basis_sets, scf_options)
        

        load = scf_options.load
        scf_options.load = "screened" #todo make calculate_three_center_integrals know that it is screening without changing load param
        for device_id in 1:num_devices
            global_device_id = device_id + (rank)*num_devices
            three_center_integrals[device_id] = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options, scf_data, global_device_id-1,num_devices_global)
        end
        scf_options.load = load

        calculate_B_GPU!(two_center_integrals, three_center_integrals, scf_data, num_devices, num_devices_global, basis_sets)
       

        #clear the memory 
        two_center_integrals = nothing
        three_center_integrals = nothing

        scf_data.gpu_data.device_fock = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_coulomb_intermediate = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_coulomb = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_exchange_intermediate = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_occupied_orbital_coefficients = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_density = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_non_zero_coefficients = Array{Array{CuArray{Float64}}}(undef, num_devices)
        scf_data.gpu_data.device_K_block = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.host_coulomb = Array{Array{Float64}}(undef, num_devices)

        scf_data.gpu_data.host_fock = Array{Array{Float64,2}}(undef, num_devices)
        scf_data.density = zeros(Float64, (scf_data.μ,scf_data.μ ))

        #println("calculating exchange block screen matrix");flush(stdout)
        calculate_exchange_block_screen_matrix(scf_data, scf_options)
        for device_id in 1:num_devices
            CUDA.device!(device_id-1)
            
            #println("creating data for fock for device $device_id");flush(stdout)

            global_device_id = device_id + (rank)*num_devices
            Q = scf_data.gpu_data.device_Q_range_lengths[global_device_id]
            scf_data.gpu_data.device_fock[device_id] = CUDA.zeros(Float64,(scf_data.μ, scf_data.μ))
            scf_data.gpu_data.device_coulomb_intermediate[device_id] = CUDA.zeros(Float64,(Q))
            scf_data.gpu_data.device_occupied_orbital_coefficients[device_id] = CUDA.zeros(Float64, (scf_data.occ, scf_data.μ))
            scf_data.gpu_data.device_coulomb[device_id] = CUDA.zeros(Float64, scf_data.screening_data.screened_indices_count)
            scf_data.gpu_data.device_density[device_id] = CUDA.zeros(Float64, (scf_data.screening_data.screened_indices_count))
            scf_data.gpu_data.device_exchange_intermediate[device_id] =  CUDA.zeros(Float64, (Q, n_ooc, p))
            scf_data.gpu_data.device_K_block[device_id] = CUDA.zeros(Float64, (scf_data.screening_data.K_block_width, scf_data.screening_data.K_block_width))
            scf_data.gpu_data.device_non_zero_coefficients[device_id] = CUDA.zeros(Float64, n_ooc, p, p)

            #host density until I can figure out how to write a kernel for copying to the screened vector on the gpu
            scf_data.density_array = zeros(Float64, (scf_data.screening_data.screened_indices_count))
            
            
            #host fock for transfering in parallel from the GPUs
            scf_data.gpu_data.host_fock[device_id] = zeros(Float64, (scf_data.μ, scf_data.μ))
            scf_data.gpu_data.host_coulomb[device_id] = zeros(Float64, scf_data.screening_data.screened_indices_count)
           


        end

    end

    
    BLAS.gemm!('T', 'N', 1.0, occupied_orbital_coefficients, occupied_orbital_coefficients, 0.0, scf_data.density)
    copy_screened_density_to_array(scf_data)

    Threads.@sync for device_id in 1:num_devices
        Threads.@spawn begin
            CUDA.device!(device_id-1)
            #println("stariting fock build for device $device_id");flush(stdout)

            global_device_id = device_id + (rank)*num_devices
            Q_length = scf_data.gpu_data.device_Q_range_lengths[global_device_id]

            ooc = scf_data.gpu_data.device_occupied_orbital_coefficients[device_id]
            density = scf_data.gpu_data.device_density[device_id]
            B = scf_data.gpu_data.device_B[device_id]
            V = scf_data.gpu_data.device_coulomb_intermediate[device_id]
            W = scf_data.gpu_data.device_exchange_intermediate[device_id]
            J = scf_data.gpu_data.device_coulomb[device_id]
            host_J = scf_data.gpu_data.host_coulomb[device_id]
            fock = scf_data.gpu_data.device_fock[device_id]
            host_fock = scf_data.gpu_data.host_fock[device_id]
            CUDA.copyto!(ooc, occupied_orbital_coefficients)

            

            #host density until I can figure out how to write a kernel for copying to the screened vector on the gpu
            CUDA.copyto!(density, scf_data.density_array)

            CUBLAS.gemv!('N', 1.0, B, density, 0.0, V)
            CUBLAS.gemv!('T', 2.0, B, V, 0.0, J)
            CUDA.copyto!(host_J, J)

            calculate_W_screened_GPU(device_id, scf_data)
            CUBLAS.gemm!('T', 'N', -1.0, reshape(W, (n_ooc * Q_length, p)), reshape(W, (n_ooc * Q_length, p)), 0.0, fock)
            CUDA.copyto!(host_fock, fock)


            # calculate_K_lower_diagonal_block_no_screen_GPU(host_fock, fock, W, Q_length, device_id, scf_data, scf_options)         
            #only the lower triangle of the GPU is calculated so we need copy the values to the upper triangle
            # for i in 1:scf_data.μ #this could be done outside of this loop and more parallel
            #     for j in 1:i-1
            #         host_fock[j, i] = scf_data.gpu_data.host_fock[device_id][i, j]
            #     end
            # end           
        end        
    end


    scf_data.two_electron_fock .= scf_data.gpu_data.host_fock[1]
    # copy_screened_coulomb_to_fock!(scf_data, scf_data.gpu_data.host_coulomb[1], scf_data.two_electron_fock)
    for device_id in 2:num_devices
        axpy!(1.0, scf_data.gpu_data.host_fock[device_id], scf_data.two_electron_fock)
        # copy_screened_coulomb_to_fock!(scf_data, scf_data.gpu_data.host_coulomb[device_id], scf_data.two_electron_fock)
    end


    println("host J:")
    println("fock: ")
    display(scf_data.two_electron_fock[1:5,1:5])
end

function calculate_W_screened_GPU(device_id, scf_data::SCFData)
    CUDA.device!(device_id-1)
    p = scf_data.μ
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    n_threads = Threads.nthreads()
    dynamic_p = n_threads + 1
    dynamic_lock = Threads.ReentrantLock()

    M = size(scf_data.D,1)
    N = scf_data.occ
    alpha = 1.0
    beta = 0.0

    B = scf_data.gpu_data.device_B[device_id]
    W = scf_data.gpu_data.device_exchange_intermediate[device_id]

    occupied_orbital_coefficients = scf_data.gpu_data.device_occupied_orbital_coefficients[device_id]
    non_zero_coefficients = scf_data.gpu_data.device_non_zero_coefficients[device_id]

    pp = 1
    K = 1
    while pp <= p
        non_zero_r_index = 1
        for r in 1:p
            if scf_data.screening_data.basis_function_screen_matrix[r, pp]
                non_zero_coefficients[:, non_zero_r_index, pp] .= view(occupied_orbital_coefficients, :, r)
                non_zero_r_index += 1
            end
        end
        K = scf_data.screening_data.non_screened_p_indices_count[pp]
      
        A_cu = view(B, :, scf_data.screening_data.sparse_p_start_indices[pp]:
            scf_data.screening_data.sparse_p_start_indices[pp]+K-1)
        B_cu = view(non_zero_coefficients, :,1:K,pp)
        C_cu = view(W, :,:,pp)

        CUDA.CUBLAS.gemm!('N','T', alpha, A_cu, B_cu, beta, C_cu)

        pp += 1


    end

    BLAS.set_num_threads(blas_threads)
end

function calculate_K_lower_diagonal_block_no_screen_GPU(host_fock::Array{Float64,2}, fock::CuArray{Float64,2}, W::CuArray{Float64,3}, Q_length::Int, device_id, scf_data::SCFData, scf_options::SCFOptions)

    CUDA.device!(device_id-1)
    n_ooc = scf_data.occ
    p = scf_data.μ
    Q = Q_length #device Q length
    K_block_width = scf_data.screening_data.K_block_width


    transA = 'T'
    transB = 'N'
    alpha = -1.0
    beta = 0.0

    M = K_block_width
    N = K_block_width
    K = Q * n_ooc

    exchange_block = scf_data.gpu_data.device_K_block[device_id]

    lower_triangle_length = get_triangle_matrix_length(scf_options.df_exchange_block_width)
    
    index = 1
    while index <= lower_triangle_length
        pp, qq = scf_data.screening_data.exchange_batch_indexes[index]

        p_range = (pp-1)*K_block_width+1:pp*K_block_width

        q_range = (qq-1)*K_block_width+1:qq*K_block_width

        A = reshape(view(W, :,:, p_range), (Q_length * n_ooc, K_block_width))
        B = reshape(view(W, :,:, q_range), (Q_length * n_ooc, K_block_width))
        C = view(exchange_block, 1:M, 1:N)

        CUBLAS.gemm!(transA, transB, alpha, A, B, beta, C)
   
        #should probably copy the block to the host and then add it to the fock matrix on the host to avoid non-contig memory access on gpu
        fock[p_range, q_range] .+= exchange_block   
        index += 1

    end

    
    if p % scf_options.df_exchange_block_width != 0 # square blocks cover the entire pq space
        p_non_square_range = 1:p    
        #non square part that didn't fit in blocks
        q_nonsquare_range = p-(p%scf_options.df_exchange_block_width)+1:p
        
        M = p
        N = length(q_nonsquare_range)
        K = Q * n_ooc
        

        A_non_square = reshape(view(W, :,:, p_non_square_range), (Q * n_ooc, p))
        B_non_square = reshape(view(W, :,:, q_nonsquare_range), (Q * n_ooc, N))
        C_non_square = reshape(view(exchange_block, 1:M*N), (M, N))
    
        CUBLAS.gemm!(transA, transB, alpha, A_non_square, B_non_square, beta, C_non_square)
    
        fock[:, q_nonsquare_range] .+= C_non_square 
    
    end
    copyto!(host_fock, fock)
end

function calculate_B_GPU!(two_center_integrals, three_center_integrals, scf_data, num_devices, num_devices_global, basis_sets)
    COMM = MPI.COMM_WORLD
    rank = MPI.Comm_rank(COMM)
    n_ranks = MPI.Comm_size(COMM)
    pq = scf_data.screening_data.screened_indices_count
    scf_data.gpu_data.device_B = Array{CuArray{Float64}}(undef, num_devices)
    device_three_center_integrals = Array{CuArray{Float64}}(undef, num_devices)
    host_B_send_buffers = Array{Array{Float64}}(undef, num_devices)
    device_J_AB_invt = Array{CuArray{Float64}}(undef, num_devices)
    device_B_send_buffers = Array{CuArray{Float64}}(undef, num_devices)

    device_B = scf_data.gpu_data.device_B


    device_Q_indices, 
    device_rank_Q_indices, 
    device_Q_range_lengths, 
    max_device_Q_range_length  = calculate_device_ranges_GPU(scf_data, num_devices, n_ranks, basis_sets)

    scf_data.gpu_data.device_Q_range_lengths = device_Q_range_lengths
    scf_data.gpu_data.device_Q_indices = device_Q_indices



    device_id_offset = rank * num_devices
    Threads.@sync for setup_device_id in 1:num_devices
        Threads.@spawn begin
            CUDA.device!(setup_device_id-1)
            global_device_id = setup_device_id  + device_id_offset
            # buffer for J_AB_invt for each device max size needed is A*A 
            # for certain B calculations the device will only need a subset of this
            # and will ref   device!(device_id - 1)reference it with a view referencing the front of the underlying array
            device_J_AB_invt[setup_device_id] = CUDA.zeros(Float64, (scf_data.A, scf_data.A))

            # CUDA.copyto!(device_J_AB_invt[device_id], two_center_integrals)

            #todo calculate the three center integrals per device (probably could directly copy to the device while it is being calculated)
            device_three_center_integrals[setup_device_id] = CUDA.zeros(Float64, size(three_center_integrals[setup_device_id]))
            CUDA.copyto!(device_three_center_integrals[setup_device_id], three_center_integrals[setup_device_id])
            # if device_id == 1
            #    CUSOLVER.potrf!('L', device_J_AB_invt[device_id])
            #    CUSOLVER.trtri!('L', 'N', device_J_AB_invt[device_id])
            # end

            device_B[setup_device_id] = CUDA.zeros(Float64, (device_Q_range_lengths[global_device_id], pq))
            device_B_send_buffers[setup_device_id] = CUDA.zeros(Float64, (max_device_Q_range_length * pq))
            host_B_send_buffers[setup_device_id] = zeros(Float64, (max_device_Q_range_length * pq))

        end #spawn
    end
    
    LAPACK.potrf!('L', two_center_integrals)
    LAPACK.trtri!('L', 'N', two_center_integrals)

    # CUDA.copyto!(two_center_integrals, device_J_AB_invt[1]) # copy back because taking subarrays on the GPU is slow / doesn't work. Need to look into if this is possible with CUDA.jl
    
    for global_recieve_device_id in 1:num_devices_global
        #println("starting device $global_recieve_device_id");flush(stdout)
        rec_device_Q_range_length = device_Q_range_lengths[global_recieve_device_id]
        recieve_rank = (global_recieve_device_id-1) ÷ num_devices
        rank_recieve_device_id = ((global_recieve_device_id-1) % num_devices) + 1 # one indexed device id for the rank 
        array_size = rec_device_Q_range_length*pq
        Threads.@sync for r_send_device_id in 1:num_devices
            Threads.@spawn begin
                CUDA.device!(r_send_device_id-1) do 
                    global_send_device_id = r_send_device_id + device_id_offset 
                    send_device_Q_range_length = device_Q_range_lengths[global_send_device_id]
                    J_AB_invt_for_device = two_center_integrals[device_Q_indices[global_recieve_device_id],device_Q_indices[global_send_device_id]]
                    device_J_AB_inv_count = send_device_Q_range_length*rec_device_Q_range_length # total number of elements in the J_AB_invt matrix for the device
                    CUDA.copyto!(device_J_AB_invt[r_send_device_id],1,J_AB_invt_for_device,1,device_J_AB_inv_count) #copy the needed J_AB_invt data to the device 
    
                    J_AB_INV_view = reshape(
                        view(device_J_AB_invt[r_send_device_id],
                        1:device_J_AB_inv_count), (rec_device_Q_range_length,send_device_Q_range_length))
                    if global_send_device_id == global_recieve_device_id
                        CUBLAS.gemm!('N', 'N', 1.0, J_AB_INV_view, device_three_center_integrals[r_send_device_id], 1.0, device_B[rank_recieve_device_id])
                    else
                        send_B_view = view(device_B_send_buffers[r_send_device_id], 1:array_size)
                        CUBLAS.gemm!('N', 'N', 1.0, J_AB_INV_view, device_three_center_integrals[r_send_device_id],
                            0.0, reshape(send_B_view, (rec_device_Q_range_length, pq)))
                    end
                end

            end #spawn
        end #sync for 

        for send_rank in 0:n_ranks-1
            for rank_send_device_id in 1:num_devices
                
                global_send_device_id = num_devices*send_rank + rank_send_device_id
              
                # skip if the device is the same as the recieve device or if rank is not the sender or reciever
                if global_send_device_id == global_recieve_device_id || (rank != recieve_rank && rank != send_rank)
                    continue
                end

                #copy from the sending device to the host send buffer
                CUDA.device!(rank_send_device_id-1) do 
                    CUDA.copyto!(host_B_send_buffers[rank_send_device_id], 1, 
                    device_B_send_buffers[rank_send_device_id], 1, array_size)
                end
                

                send_unique_tag = send_rank*100000 + recieve_rank*10000 + global_send_device_id*1000 + global_recieve_device_id*100
                if send_rank == recieve_rank # devices belong to the same rank 
                    CUDA.device!(rank_recieve_device_id-1) do 
                        CUDA.copyto!(device_B_send_buffers[rank_recieve_device_id],
                         1, host_B_send_buffers[rank_send_device_id], 1, array_size)
                    end
                    
                elseif rank == send_rank
                    CUDA.device!(rank_send_device_id-1) do 
                        MPI.Send(host_B_send_buffers[rank_send_device_id], recieve_rank, send_unique_tag , COMM)
                    end
                elseif rank == recieve_rank
                    CUDA.device!(rank_recieve_device_id-1) do 
                        MPI.Recv!(host_B_send_buffers[rank_recieve_device_id], send_rank, send_unique_tag, COMM)
                        CUDA.copyto!(device_B_send_buffers[rank_recieve_device_id],1,
                         host_B_send_buffers[rank_recieve_device_id],1, array_size)
                    end
                end
                    
                if rank == recieve_rank # add the sent buffer to the device B matrix
                    CUDA.device!(rank_recieve_device_id-1) do 
                        device_B_send_view = reshape(view(device_B_send_buffers[rank_recieve_device_id], 1:array_size), (rec_device_Q_range_length, pq))
                        CUDA.axpy!(1.0, device_B_send_view, device_B[rank_recieve_device_id])
                    end                
                end
            end
        end
    end
    MPI.Barrier(COMM) #sync all ranks 
    return
end


# get the index data when using GPUs
# indexes go from rank 0 GPUs in order to rank N gpus in order
# E.g. if 4 GPUs per rank and 4 ranks 
# global device ID: 1, 2 ,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
# rank device ID:   1, 2, 3, 4, 1, 2, 3, 4, 1, 2,  3,  4,  1,  2,  3,  4
# rank:             0, 0, 0, 0, 1, 1, 1, 1, 2, 2,  2,  2,  3,  3,  3,  3
# with scf_data.A = 63
# ranges[1] = 1:4
# ranges[2] = 5:8
# ...
# ranges[16] = 61:63
# Currently we assume easch rank has the same number of devices
# A GPU device behaves as though it were a CPU rank in non-GPU mode
function calculate_device_ranges_GPU(scf_data, num_devices, n_ranks, basis_sets)

    load_balance_indicies = []
    total_devices = num_devices*n_ranks
    for rank in 0:total_devices-1
        push!(load_balance_indicies, static_load_rank_indicies_3_eri(rank, total_devices, basis_sets))
    end

    device_Q_indices = Array{UnitRange{Int64}}(undef, num_devices * n_ranks)
    device_rank_Q_indices = Array{UnitRange{Int64}}(undef, num_devices * n_ranks)
    indices_per_device = Array{Int}(undef, num_devices * n_ranks)

    global_device_id = 1

    max_device_Q_range_length = 0


    for rank in 0:n_ranks-1
        for device in 1:num_devices
            device_basis_indicies = load_balance_indicies[global_device_id][2]
            device_Q_indices[global_device_id] = device_basis_indicies[1]:device_basis_indicies[end]
            indices_per_device[global_device_id] = length(device_Q_indices[global_device_id])
            device_rank_Q_indices[global_device_id] = 1:indices_per_device[global_device_id]
            if indices_per_device[global_device_id] > max_device_Q_range_length
                max_device_Q_range_length = indices_per_device[global_device_id]
            end
            global_device_id += 1
        end
    end
    return device_Q_indices, device_rank_Q_indices, indices_per_device, max_device_Q_range_length
end


function free_gpu_memory(scf_data::SCFData)
    # for device_id in 1:scf_data.gpu_data.number_of_devices_used
    #     CUDA.device!(device_id-1)
    #     CUDA.unsafe_free!(scf_data.gpu_data.device_B[device_id])
    #     CUDA.unsafe_free!(scf_data.gpu_data.device_fock[device_id])
    #     CUDA.unsafe_free!(scf_data.gpu_data.device_coulomb_intermediate[device_id])
    #     CUDA.unsafe_free!(scf_data.gpu_data.device_exchange_intermediate[device_id])
    #     CUDA.unsafe_free!(scf_data.gpu_data.device_occupied_orbital_coefficients[device_id])
    #     CUDA.unsafe_free!(scf_data.gpu_data.device_density[device_id])
    # end

    # GC.gc(true) #force cleanup of the GPU data
    # for device_id in 1:scf_data.gpu_data.number_of_devices_used
    #     CUDA.device!(device_id-1)
    #     CUDA.reclaim()
    # end
end