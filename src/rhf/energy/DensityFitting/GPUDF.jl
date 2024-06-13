using CUDA
using CUDA.CUBLAS
using CUDA.CUSOLVER
using LinearAlgebra
using Base.Threads

function df_rhf_fock_build_GPU!(scf_data, jeri_engine_thread_df::Vector{T}, jeri_engine_thread::Vector{T2},
    basis_sets::CalculationBasisSets,
    occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine,T2<:RHFTEIEngine}
    comm = MPI.COMM_WORLD
    pq = scf_data.μ^2

    p = scf_data.μ
    n_ooc = scf_data.occ

    devices = CUDA.devices()
    num_devices = 1
    scf_data.gpu_data.number_of_devices_used = num_devices
    occupied_orbital_coefficients = permutedims(occupied_orbital_coefficients, (2,1))


    device!(0)
    if iteration == 1
        two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
        get_screening_metadata!(scf_data, scf_options.df_screening_sigma, jeri_engine_thread, two_center_integrals, basis_sets, scf_options)
        

        load = scf_options.load
        scf_options.load = "screened" #todo make calculate_three_center_integrals know that it is screening without changing load param
        three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options, scf_data)
        scf_options.load = load

        device_three_center_integrals = CUDA.zeros(Float64, size(three_center_integrals))
        CUDA.copyto!(device_three_center_integrals, three_center_integrals)

        device_id = 1

        device_J_AB_invt = Array{CuArray{Float64}}(undef, num_devices)
        device_J_AB_invt[1] = CUDA.zeros(Float64, size(two_center_integrals))
        CUDA.copyto!(device_J_AB_invt[1], two_center_integrals)

        CUSOLVER.potrf!('L', device_J_AB_invt[device_id])
        #is this one necessary?
        CUSOLVER.trtri!('L', 'N', device_J_AB_invt[device_id])
        scf_data.gpu_data.device_B =  Array{CuArray{Float64}}(undef, num_devices)
        
        scf_data.gpu_data.device_B[1] = CUDA.zeros(Float64, size(three_center_integrals))

        CUDA.CUBLAS.trmm!('L', 'L', 'N', 'N', 1.0, device_J_AB_invt[device_id], device_three_center_integrals, scf_data.gpu_data.device_B[1])    

        # calculate_B_GPU!(two_center_integrals, three_center_integrals, scf_data, num_devices)

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

        scf_data.gpu_data.host_coulomb = Array{Array{Float64}}(undef, num_devices)

        scf_data.gpu_data.host_fock = Array{Array{Float64,2}}(undef, num_devices)
        scf_data.density = zeros(Float64, (scf_data.μ,scf_data.μ ))
        
        for device_id in 1:num_devices
            device!(device_id - 1)
            Q = scf_data.A
            scf_data.gpu_data.device_fock[device_id] = CUDA.zeros(Float64,(scf_data.μ, scf_data.μ))
            scf_data.gpu_data.device_coulomb_intermediate[device_id] = CUDA.zeros(Float64,(Q))
            scf_data.gpu_data.device_exchange_intermediate[device_id] =
                CUDA.zeros(Float64, (Q, n_ooc, p))
            scf_data.gpu_data.device_occupied_orbital_coefficients[device_id] = CUDA.zeros(Float64, (scf_data.occ, scf_data.μ))
            scf_data.gpu_data.device_coulomb[device_id] = CUDA.zeros(Float64, scf_data.screening_data.screened_indices_count)
            scf_data.gpu_data.device_density[device_id] = CUDA.zeros(Float64, (scf_data.screening_data.screened_indices_count))

            #host density until I can figure out how to write a kernel for copying to the screened vector on the gpu
            scf_data.density_array = zeros(Float64, (scf_data.screening_data.screened_indices_count))
            
            #host fock for transfering in parallel from the GPUs
            scf_data.gpu_data.host_fock[device_id] = zeros(Float64, (scf_data.μ, scf_data.μ))
            scf_data.gpu_data.device_non_zero_coefficients[device_id] = CUDA.zeros(Float64, n_ooc, p, p)
            scf_data.gpu_data.host_coulomb[device_id] = zeros(Float64, scf_data.screening_data.screened_indices_count)
            


        end
    end


    Threads.@sync for device_id in 1:num_devices
        Threads.@spawn begin
            device!(device_id - 1)
            ooc = scf_data.gpu_data.device_occupied_orbital_coefficients[device_id]
            density = scf_data.gpu_data.device_density[device_id]
            B = scf_data.gpu_data.device_B[device_id]
            Q_length = scf_data.A #scf_data.gpu_data.device_Q_range_lengths[device_id]
            V = scf_data.gpu_data.device_coulomb_intermediate[device_id]
            W = scf_data.gpu_data.device_exchange_intermediate[device_id]
            J = scf_data.gpu_data.device_coulomb[device_id]
            host_J = scf_data.gpu_data.host_coulomb[device_id]
            fock = scf_data.gpu_data.device_fock[device_id]
            host_density = scf_data.density_array
            copyto!(ooc, occupied_orbital_coefficients)

            #host density until I can figure out how to write a kernel for copying to the screened vector on the gpu
            BLAS.gemm!('T', 'N', 1.0, occupied_orbital_coefficients, occupied_orbital_coefficients, 0.0, scf_data.density)
            copy_screened_density_to_array(scf_data)
            copyto!(density, scf_data.density_array)

            #GPU contractions 
            CUBLAS.gemv!('N', 1.0, B, density, 0.0, V)
            CUBLAS.gemv!('T', 2.0, B, V, 0.0, J)

            copyto!(host_J, J)

            calculate_W_screened_GPU(scf_data)
            calculate_K_lower_diagonal_block_no_screen(fock, W, Q_length, scf_data, scf_options)

           
            copyto!(scf_data.gpu_data.host_fock[device_id], fock)
            
            #only the lower triangle of the GPU is calculated so we need copy the values to the upper triangle
            for i in 1:scf_data.μ #this could be done outside of this loop and more parallel
                for j in 1:i-1
                    scf_data.gpu_data.host_fock[device_id][j, i] = scf_data.gpu_data.host_fock[device_id][i, j]
                end
            end           
        end
    end


    scf_data.two_electron_fock .= scf_data.gpu_data.host_fock[1]
    for device_id in 2:num_devices
        axpy!(1.0, scf_data.gpu_data.host_fock[device_id], scf_data.two_electron_fock)
        copy_screened_coulomb_to_fock!(scf_data, scf_data.gpu_data.host_coulomb[device_id], scf_data.two_electron_fock)
    end

end

function calculate_W_screened_GPU(scf_data::SCFData)
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

    B = scf_data.gpu_data.device_B[1]
    W = scf_data.gpu_data.device_exchange_intermediate[1]

    linear_indicesB = LinearIndices(B)
    linear_indicesW = LinearIndices(W)

    occupied_orbital_coefficients = scf_data.gpu_data.device_occupied_orbital_coefficients[1]
    non_zero_coefficients = scf_data.gpu_data.device_non_zero_coefficients[1]

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
        # A_ptr = pointer(scf_data.D, linear_indicesB[1, scf_data.screening_data.sparse_p_start_indices[pp]])
        # B_ptr = pointer(non_zero_coefficients[pp], 1)
        # C_ptr = pointer(scf_data.D_tilde, linear_indicesW[1, 1, pp])

        A_cu = view(B, :, scf_data.screening_data.sparse_p_start_indices[pp]:
            scf_data.screening_data.sparse_p_start_indices[pp]+K-1)
        B_cu = view(non_zero_coefficients, :,1:K,pp)
        C_cu = view(W, :,:,pp)

        CUDA.CUBLAS.gemm!('N','T', alpha, A_cu, B_cu, beta, C_cu)

        pp += 1


    end
    BLAS.set_num_threads(blas_threads)
end

function calculate_K_lower_diagonal_block_no_screen(fock::CuArray{Float64,2}, W::CuArray{Float64,3}, Q_length::Int, scf_data::SCFData, scf_options::SCFOptions)
    n_ooc = scf_data.occ
    p = scf_data.μ
    Q = Q_length #device Q length
    K_block_width = scf_data.screening_data.K_block_width

    transA = 'T'
    transB = 'N'
    alpha = -1.0
    beta = 0.0
    linear_indices = LinearIndices(W)

    M = K_block_width
    N = K_block_width
    K = Q * n_ooc

    exchange_block = CUDA.zeros(Float64, (M, N))

    K_linear_indices = LinearIndices((M,N))
    

    lower_triangle_length = get_triangle_matrix_length(scf_options.df_exchange_block_width)
    
    calculate_exchange_block_screen_matrix(scf_data, scf_options)

    index = 1
    host_block = zeros(M, N)
    while index <= lower_triangle_length
        pp, qq = scf_data.screening_data.exchange_batch_indexes[index]

        p_range = (pp-1)*K_block_width+1:pp*K_block_width
        p_start = (pp - 1) * K_block_width + 1

        q_range = (qq-1)*K_block_width+1:qq*K_block_width
        q_start = (qq - 1) * K_block_width + 1

        A = reshape(view(W, :,:, p_range), (Q_length * n_ooc, K_block_width))
        B = reshape(view(W, :,:, q_range), (Q_length * n_ooc, K_block_width))
        C = exchange_block

        CUBLAS.gemm!(transA, transB, alpha, A, B, beta, C)

        copyto!(host_block, exchange_block)
        # println("host block")
        
        fock[p_range, q_range] .+= exchange_block




        index += 1
        # if pp != qq
        #     fock[q_range, p_range] .+= transpose(exchange_block) #probably worth doing this on the CPU instead transpose is slow on GPU 
        # end
    end

    if p % scf_options.df_exchange_block_width == 0 # square blocks cover the entire pq space
        return
    end

    p_non_square_range = 1:p
    p_non_square_start = 1

    #non square part 
    q_nonsquare_range = p-(p%scf_options.df_exchange_block_width)+1:p
    q_nonsquare_start = q_nonsquare_range[1]
    
    M = p
    N = length(q_nonsquare_range)
    K = Q * n_ooc
    

    A_nonsquare_ptr = CUDA.pointer(W, linear_indices[1, 1, p_non_square_start])
    B_nonsquare_ptr = CUDA.pointer(W, linear_indices[1, 1, q_nonsquare_start])
    C_nonsquare_ptr = CUDA.pointer(exchange_block, 1)

    CUBLAS.gemm!(transA, transB, alpha, A_nonsquare_ptr, B_nonsquare_ptr, beta, C_nonsquare_ptr)
    
    non_square_buffer = reshape(view(scf_data.k_blocks, 1:M*N), (p, length(q_nonsquare_range))) 

    scf_data.two_electron_fock[p_non_square_range, q_nonsquare_range] .= non_square_buffer

    # CUBLAS.gemm!('T', 'N', -1.0, reshape(W, (n_ooc * Q_length, p)), reshape(W, (n_ooc * Q_length, p)), 1.0, fock)

end

function calculate_B_GPU!(two_center_integrals, three_center_integrals, scf_data, num_devices)
    pq = scf_data.μ^2

    device_J_AB_invt = Array{CuArray{Float64}}(undef, num_devices)
    scf_data.gpu_data.device_B = Array{CuArray{Float64}}(undef, num_devices)
    device_B = scf_data.gpu_data.device_B
    device_three_center_integrals = Array{CuArray{Float64}}(undef, num_devices)
    scf_data.gpu_data.device_B_send_buffers = Array{CuArray{Float64}}(undef, num_devices)

    device_B_send_buffers = scf_data.gpu_data.device_B_send_buffers
    indices_per_device,
    device_Q_range_starts,
    device_Q_range_ends,
    device_Q_indices,
    device_Q_range_lengths,
    max_device_Q_range_length = calculate_device_ranges_GPU(scf_data, num_devices)

    scf_data.gpu_data.device_Q_range_lengths = device_Q_range_lengths
    scf_data.gpu_data.device_Q_range_starts = device_Q_range_starts
    scf_data.gpu_data.device_Q_range_ends = device_Q_range_ends
    scf_data.gpu_data.device_Q_indices = device_Q_indices


    three_eri_linear_indices = LinearIndices(three_center_integrals)

    Threads.@sync for device_id in 1:num_devices
        Threads.@spawn begin
            device!(device_id - 1)
            # buffer for J_AB_invt for each device max size needed is A*A 
            # for certain B calculations the device will only need a subset of this
            # and will reference it with a view referencing the front of the underlying array
            device_J_AB_invt[device_id] = CUDA.zeros(Float64, (scf_data.A, scf_data.A))
            device_three_center_integrals[device_id] = CUDA.zeros(Float64, (scf_data.μ, scf_data.μ, device_Q_range_lengths[device_id]))

            copyto!(device_J_AB_invt[device_id], two_center_integrals)
            three_center_integral_size = device_Q_range_lengths[device_id] * pq
            device_pointer = pointer(device_three_center_integrals[device_id], 1)
            device_pointer_start = three_eri_linear_indices[1, 1, device_Q_range_starts[device_id]]
            host_pointer = pointer(three_center_integrals, device_pointer_start)
            unsafe_copyto!(device_pointer, host_pointer, three_center_integral_size)

            
            CUSOLVER.potrf!('L', device_J_AB_invt[device_id])
            CUSOLVER.trtri!('L', 'N', device_J_AB_invt[device_id])

            device_B[device_id] = CUDA.zeros(Float64, (device_Q_range_lengths[device_id], scf_data.μ, scf_data.μ))
            device_B_send_buffers[device_id] = CUDA.zeros(Float64, (max_device_Q_range_length * scf_data.μ * scf_data.μ))
            
        end
    end


    device!(0)
    CUDA.copyto!(two_center_integrals, device_J_AB_invt[1]) # copy back because taking subarrays on the GPU is slow / doesn't work. Need to look into if this is possible with CUDA.jl
    
    J_AB_INV = two_center_integrals #simple rename to avoid confusion 

    for recieve_device_id in 1:num_devices
        device!(recieve_device_id - 1)
        rec_device_Q_range_length = device_Q_range_lengths[recieve_device_id]

        Threads.@sync for send_device_id in 1:num_devices
            Threads.@spawn begin
                device!(send_device_id - 1)
                send_device_Q_range_length = device_Q_range_lengths[send_device_id]
                J_AB_invt_for_device = J_AB_INV[device_Q_indices[recieve_device_id], device_Q_indices[send_device_id]]

                device_J_AB_inv_count = rec_device_Q_range_length * send_device_Q_range_length # total number of elements in the J_AB_invt matrix for the device


                three_eri_view = reshape(device_three_center_integrals[send_device_id], (pq, send_device_Q_range_length))
                J_AB_INV_view = reshape(view(device_J_AB_invt[send_device_id], 1:device_J_AB_inv_count), (rec_device_Q_range_length, send_device_Q_range_length))

                CUDA.copyto!(device_J_AB_invt[send_device_id], J_AB_invt_for_device) #copy the needed J_AB_invt data to the device 

                if send_device_id != recieve_device_id
                    send_B_view = view(device_B_send_buffers[send_device_id], 1:rec_device_Q_range_length*pq)
                    CUBLAS.gemm!('N', 'T', 1.0, J_AB_INV_view, three_eri_view,
                        0.0, reshape(send_B_view, (rec_device_Q_range_length, pq)))
                else
                    CUBLAS.gemm!('N', 'T', 1.0, J_AB_INV_view, three_eri_view, 1.0,
                        reshape(device_B[recieve_device_id], (rec_device_Q_range_length, pq)))
                end

            end
        end


        device!(recieve_device_id - 1)
        array_size = rec_device_Q_range_length * pq
        data = zeros(array_size)
        for send_device_id in 1:num_devices
            if send_device_id == recieve_device_id
                continue
            end
            device!(send_device_id - 1)

            CUDA.unsafe_copyto!(data, 1, device_B_send_buffers[send_device_id], 1, array_size)
            
            device!(recieve_device_id - 1)
            CUDA.unsafe_copyto!(device_B_send_buffers[recieve_device_id], 1, data, 1, array_size)
            
            CUBLAS.axpy!(array_size, 1.0,
                reshape(view(device_B_send_buffers[recieve_device_id], 1:array_size), (array_size)),
                reshape(view(device_B[recieve_device_id], 1:array_size), (array_size)))
        end
        
    end
    for device_id in 1:num_devices
        CUDA.unsafe_free!(device_J_AB_invt[device_id])
        CUDA.unsafe_free!(device_three_center_integrals[device_id])
        CUDA.unsafe_free!(device_B_send_buffers[device_id])
    end
    
    GC.gc(true) #force cleanup of the GPU data
    for device_id in 1:scf_data.gpu_data.number_of_devices_used
        CUDA.device!(device_id - 1)
        CUDA.reclaim()
    end
    return
end

function calculate_device_ranges_GPU(scf_data, num_devices)
    indices_per_device = scf_data.A ÷ num_devices
    device_Q_range_starts = 1:indices_per_device+1:scf_data.A
    device_Q_range_ends = device_Q_range_starts .+ indices_per_device

    device_Q_indices = [device_Q_range_starts[i]:device_Q_range_ends[i] for i in 1:num_devices]
    device_Q_indices[end] = device_Q_range_starts[end]:scf_data.A
    device_Q_range_lengths = length.(device_Q_indices)
    max_device_Q_range_length = maximum(device_Q_range_lengths)
    return indices_per_device, device_Q_range_starts, device_Q_range_ends, device_Q_indices, device_Q_range_lengths, max_device_Q_range_length
end


function free_gpu_memory(scf_data::SCFData)
    for device_id in 1:scf_data.gpu_data.number_of_devices_used
        CUDA.device!(device_id - 1)
        CUDA.unsafe_free!(scf_data.gpu_data.device_B[device_id])
        CUDA.unsafe_free!(scf_data.gpu_data.device_fock[device_id])
        CUDA.unsafe_free!(scf_data.gpu_data.device_coulomb_intermediate[device_id])
        CUDA.unsafe_free!(scf_data.gpu_data.device_exchange_intermediate[device_id])
        CUDA.unsafe_free!(scf_data.gpu_data.device_occupied_orbital_coefficients[device_id])
        CUDA.unsafe_free!(scf_data.gpu_data.device_density[device_id])
    end

    GC.gc(true) #force cleanup of the GPU data
    for device_id in 1:scf_data.gpu_data.number_of_devices_used
        CUDA.device!(device_id - 1)
        CUDA.reclaim()
    end
    println("done freeing up GPU memory")
end