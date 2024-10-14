using MPI
using CUDA
using CUDA.CUBLAS
using CUDA.CUSOLVER
using LinearAlgebra
using Base.Threads
using HDF5
using JuliaChem.Shared.JCTC

function df_rhf_fock_build_GPU!(scf_data, jeri_engine_thread_df::Vector{T}, jeri_engine_thread::Vector{T2},
    basis_sets::CalculationBasisSets,
    occupied_orbital_coefficients, iteration, scf_options::SCFOptions, H::Array{Float64},
    jc_timing::JCTiming) where {T<:DFRHFTEIEngine,T2<:RHFTEIEngine}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)

    p = scf_data.μ
    n_ooc = scf_data.occ

    devices = CUDA.devices()
    #get environment variable for number of devices to use
    num_devices = 1
    if haskey(ENV, "JC_NUM_DEVICES")
        num_devices = parse(Int64, ENV["JC_NUM_DEVICES"])
    end

    # num_devices = Int64(length(devices))
    num_devices_global = num_devices*n_ranks  
    scf_data.gpu_data.number_of_devices_used = num_devices
    occupied_orbital_coefficients = permutedims(occupied_orbital_coefficients, (2,1))


    n_j_streams_per_device = 20

    three_center_integrals = Array{Array{Float64}}(undef, num_devices)
    use_K_rect = false
    if haskey(ENV, "JC_USE_K_RECT")
        use_K_rect = parse(Bool, ENV["JC_USE_K_RECT"])
    end
    if iteration == 1
        scf_options.df_exchange_block_width = 1
        #get env variable to set the block width or not 
        do_K_sym = false
        if haskey(ENV, "JC_DF_DO_K_SYM")
            do_K_sym = parse(Bool, ENV["JC_DF_DO_K_SYM"])
        end
        do_adaptive = false
        if haskey(ENV, "JC_USE_ADAPTIVE")
            do_adaptive = parse(Bool, ENV["JC_USE_ADAPTIVE"])
        end
        #k rect
        
        if do_K_sym || do_adaptive    
            if p >= 1600
                scf_options.df_exchange_block_width = 12
            elseif p >= 1300
                scf_options.df_exchange_block_width = 8
            elseif p >= 1200
                scf_options.df_exchange_block_width = 4
            elseif p > 800 || do_K_sym #if do_K_sym is true then we are using the symmetric algorithm default to 2 for small systems
                scf_options.df_exchange_block_width = 2
            end 
        end

        two_eri_time = @elapsed two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
        screening_time = @elapsed begin
            get_screening_metadata!(scf_data, scf_options.df_screening_sigma, jeri_engine_thread, two_center_integrals, basis_sets, jc_timing)
            calculate_exchange_block_screen_matrix(scf_data, scf_options,jc_timing)
        end
        load = scf_options.load
        scf_options.load = "screened" #todo make calculate_three_center_integrals know that it is screening without changing load param
        
        three_eri_time = @elapsed begin
            for device_id in 1:num_devices #the method being called uses many threads do not need to thread by device
                global_device_id = device_id + (rank)*num_devices
                three_center_integrals[device_id] = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options, scf_data, global_device_id-1,num_devices_global)
            end
        end
        scf_options.load = load

        calculate_B_GPU!(two_center_integrals, three_center_integrals, scf_data, num_devices, num_devices_global, basis_sets, jc_timing)
       

        #clear the memory 
        two_center_integrals = nothing
        three_center_integrals = nothing

        scf_data.gpu_data.device_fock = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_coulomb_intermediate = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_coulomb = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_stream_coulmob = Array{Array{CuArray{Float64}}}(undef, num_devices)
        scf_data.gpu_data.device_stream_coulmob_intermediate = Array{Array{CuArray{Float64}}}(undef, num_devices)


        scf_data.gpu_data.device_exchange_intermediate = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_occupied_orbital_coefficients = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_density = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_screened_density = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_non_zero_coefficients = Array{Array{CuArray{Float64}}}(undef, num_devices)
        scf_data.gpu_data.device_K_block = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_non_square_K_block = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.host_coulomb = Array{Array{Float64,1}}(undef, num_devices)

        scf_data.gpu_data.device_range_p = Array{CuArray{Int64,1}}(undef, num_devices)
        scf_data.gpu_data.device_range_start = Array{CuArray{Int64,1}}(undef, num_devices)
        scf_data.gpu_data.device_range_end = Array{CuArray{Int64,1}}(undef, num_devices)
        scf_data.gpu_data.device_range_sparse_start = Array{CuArray{Int64,1}}(undef, num_devices)
        scf_data.gpu_data.device_range_sparse_end = Array{CuArray{Int64,1}}(undef, num_devices)
        scf_data.gpu_data.device_sparse_to_p = Array{CuArray{Int64,1}}(undef, num_devices)
        scf_data.gpu_data.device_sparse_to_q = Array{CuArray{Int64,1}}(undef, num_devices)

        scf_data.gpu_data.sparse_pq_index_map = Array{CuArray{Int64,2}}(undef, num_devices)
        
        scf_data.gpu_data.host_fock = Array{Array{Float64,2}}(undef, num_devices)
        scf_data.density = zeros(Float64, (scf_data.μ,scf_data.μ ))

        scf_data.screening_data.non_zero_coefficients = zeros(Float64, n_ooc, p, p)

        
        Threads.@threads for device_id in 1:num_devices
            CUDA.device!(device_id-1)

            #device host data 
            global_device_id = device_id + (rank)*num_devices
            Q = scf_data.gpu_data.device_Q_range_lengths[global_device_id]
            scf_data.gpu_data.host_coulomb[device_id] = zeros(Float64, scf_data.screening_data.screened_indices_count)

            #host density until I can figure out how to write a kernel for copying to the screened vector on the gpu
            scf_data.density_array = zeros(Float64, (scf_data.screening_data.screened_indices_count))
            #host fock for transfering in parallel from the GPUs
            scf_data.gpu_data.host_fock[device_id] = zeros(Float64, (scf_data.μ, scf_data.μ))
            scf_data.gpu_data.host_coulomb[device_id] = zeros(Float64, scf_data.screening_data.screened_indices_count)

            #cuda device data
            scf_data.gpu_data.device_fock[device_id] = CUDA.zeros(Float64,(scf_data.μ, scf_data.μ))

            scf_data.gpu_data.device_density[device_id] = CUDA.zeros(Float64, (p,p))
            scf_data.gpu_data.device_screened_density[device_id] = CUDA.zeros(Float64, (scf_data.screening_data.screened_indices_count))
            scf_data.gpu_data.device_coulomb[device_id] = CUDA.zeros(Float64, scf_data.screening_data.screened_indices_count)
            scf_data.gpu_data.device_coulomb_intermediate[device_id] = CUDA.zeros(Float64,(Q))
            scf_data.gpu_data.device_stream_coulmob[device_id] = Array{CuArray{Float64}}(undef, n_j_streams_per_device)
            scf_data.gpu_data.device_stream_coulmob_intermediate[device_id] = Array{CuArray{Float64}}(undef, n_j_streams_per_device)
            
            for stream_id in 1:n_j_streams_per_device
                scf_data.gpu_data.device_stream_coulmob[device_id][stream_id] = CUDA.zeros(Float64, (scf_data.screening_data.screened_indices_count))
                scf_data.gpu_data.device_stream_coulmob_intermediate[device_id][stream_id] = CUDA.zeros(Float64, (Q))
            end

            scf_data.gpu_data.device_occupied_orbital_coefficients[device_id] = CUDA.zeros(Float64, (scf_data.occ, scf_data.μ))
            scf_data.gpu_data.device_non_zero_coefficients[device_id] = CUDA.zeros(Float64, n_ooc, p, p)
            scf_data.gpu_data.device_exchange_intermediate[device_id] =  CUDA.zeros(Float64, (Q, n_ooc, p))
            lower_triangle_length = get_triangle_matrix_length(scf_options.df_exchange_block_width)#should only be done on first iteration 
            scf_data.gpu_data.device_K_block[device_id] = CUDA.zeros(Float64, (scf_data.screening_data.K_block_width, scf_data.screening_data.K_block_width, lower_triangle_length))
            
            ################   duplicated logic! move this to a shared place   ##########################
            row_nonsquare_range = p-(p%scf_options.df_exchange_block_width)+1:p
            scf_data.gpu_data.device_non_square_K_block[device_id] = CUDA.zeros(Float64, (length(row_nonsquare_range), p))
            ############################################################################################################
            
            if rank == 0 && device_id == 1
                scf_data.gpu_data.device_H = CUDA.zeros(Float64, (scf_data.μ, scf_data.μ))
                CUDA.copyto!(scf_data.gpu_data.device_H, H)
            end
        end
        gpu_screening_setup = @elapsed setup_gpu_screening_data!(scf_data, num_devices)


        jc_timing.non_timing_data[JCTC.contraction_algorithm] = "screened gpu"
        jc_timing.timings[JCTC.two_eri_time] = two_eri_time
        jc_timing.timings[JCTC.three_eri_time] = three_eri_time
        jc_timing.timings[JCTC.screening_time] = screening_time
        jc_timing.timings[JCTC.B_time] = B_time
        jc_timing.timings[JCTC.GPU_B_time] = B_time
        jc_timing.timings[JCTC.GPU_screening_setup_time] = gpu_screening_setup
        jc_timing.non_timing_data[JCTC.GPU_num_devices] = string(num_devices)
    end

  


    V_times = zeros(Float64, num_devices)
    J_times = zeros(Float64, num_devices)

    W_times = zeros(Float64, num_devices)
    K_times = zeros(Float64, num_devices)
    H_times = zeros(Float64, num_devices)
    density_times = zeros(Float64, num_devices)
    gpu_fock_times = zeros(Float64, num_devices)
    non_zero_coeff_times  = zeros(Float64, num_devices)
    device_start = zeros(Float64, num_devices)
    device_time = zeros(Float64, num_devices)
    fock_copy_time = 0.0


    n_threads = Threads.nthreads()
    threads_per_device = Int64((n_threads - num_devices) ÷ num_devices) 
    start_all = time()
    
    total_fock_gpu_time = @elapsed begin 
        Threads.@sync for device_id in 1:num_devices
            Threads.@spawn begin
                gpu_fock_times[device_id] = @elapsed begin 
                    CUDA.device!(device_id-1)
                    global_device_id = device_id + (rank)*num_devices
                    Q_length = scf_data.gpu_data.device_Q_range_lengths[global_device_id]

                    ooc = scf_data.gpu_data.device_occupied_orbital_coefficients[device_id]
                    density = scf_data.gpu_data.device_screened_density[device_id]
                    B = scf_data.gpu_data.device_B[device_id]
                    V = scf_data.gpu_data.device_coulomb_intermediate[device_id]
                    W = scf_data.gpu_data.device_exchange_intermediate[device_id]
                    J = scf_data.gpu_data.device_coulomb[device_id]
                    fock = scf_data.gpu_data.device_fock[device_id]
                    host_fock = scf_data.gpu_data.host_fock[device_id]
                
                
                    CUDA.copyto!(ooc, occupied_orbital_coefficients)
                    CUDA.synchronize()
                    non_zero_coeff_times[device_id] = @elapsed form_nozero_coefficient_matrix!(scf_data, device_id)

                    W_times[device_id]  = @elapsed calculate_W_screened_GPU(device_id, scf_data, threads_per_device)
                    if use_K_rect 
                        K_times[device_id]  = @elapsed calculate_K_upper_diagonal_rectangle_blocks(fock, W, Q_length, device_id,
                        scf_data, scf_options, threads_per_device)
                    elseif scf_options.df_exchange_block_width != 1 
                        lower_triangle_length = get_triangle_matrix_length(scf_options.df_exchange_block_width)#should only be done on first iteration 
                        K_times[device_id]  = @elapsed calculate_K_lower_diagonal_block_no_screen_GPU(host_fock, fock, W, Q_length, device_id,
                        scf_data, scf_options, lower_triangle_length, threads_per_device)       
                    else
                        K_times[device_id]  = @elapsed calcululate_K_no_sym_GPU!(fock, W, p, scf_data.occ, Q_length, device_id)
                    end
                    H_times[device_id]  = @elapsed begin 
                        if rank == 0 && device_id == 1
                            CUDA.axpy!(1.0, scf_data.gpu_data.device_H, fock)
                        end
                    end
                    CUDA.synchronize()
                    density_times[device_id]  = @elapsed form_screened_density!(scf_data, device_id)
                    V_times[device_id]  = @elapsed calculate_V_screened_GPU(density, B, density)
                    J_times[device_id]  = @elapsed calculate_J_screened_GPU(J, B, V)
                    # J_times = @elapsed calculate_J_screened_symmetric_GPU(density, host_J, J, V, B, scf_data, device_id, n_j_streams_per_device)
                    # calculate_J_screened_symmetric_GPU(density, host_J, J, V, B, scf_data, device_id, n_j_streams_per_device)
                
                    # J_times = @elapsed calculate_J_screened_symmetric_GPU(density, host_J, J, V, B, scf_data, device_id, n_j_streams_per_device)
                    # CUDA.synchronize()

                    gpu_copy_J[device_id] = @elapsed begin 
                        numblocks = ceil(Int64, scf_data.screening_data.screened_indices_count/256)
                        threads = min(256, scf_data.screening_data.screened_indices_count)
    
                        device_sparse_to_p = scf_data.gpu_data.device_sparse_to_p[device_id]
                        device_sparse_to_q = scf_data.gpu_data.device_sparse_to_q[device_id]
    
                        @cuda threads=threads blocks=numblocks copy_screened_J_to_fock_upper_triangle(fock, J, device_sparse_to_p, 
                            device_sparse_to_q, scf_data.screening_data.screened_indices_count)
                        CUDA.synchronize() 
                    end
                  
                    gpu_copy_sym_time[device_id] = @elapsed begin
                        numblocks = ceil(Int64, scf_data.screening_data.screened_indices_count/256)
                        threads = min(256, scf_data.screening_data.screened_indices_count)
    
                        device_sparse_to_p = scf_data.gpu_data.device_sparse_to_p[device_id]
                        device_sparse_to_q = scf_data.gpu_data.device_sparse_to_q[device_id]
    
                        @cuda threads=threads blocks=numblocks copy_screened_J_to_fock_symmetric(fock, J, device_sparse_to_p, 
                            device_sparse_to_q, scf_data.screening_data.screened_indices_count)
                        CUDA.synchronize() 
                    end
                end # gpu fock time elapsed
            end #spawn     
        end#sync

        fock_copy_time = @elapsed begin
            Threads.@threads for device_id in 1:num_devices
                CUDA.device!(device_id-1)
                CUDA.copyto!(scf_data.gpu_data.host_fock[device_id], scf_data.gpu_data.device_fock[device_id])  
                CUDA.synchronize()
            end
            scf_data.two_electron_fock = scf_data.gpu_data.host_fock[1]
            for device_id in 2:num_devices
                axpy!(1.0, scf_data.gpu_data.host_fock[device_id], scf_data.two_electron_fock)
            end 
        end #copy_time elapsed 
    end# total fock gpu time elapsed



    for device_id in 1:num_devices
        jc_timing.timings[JCTiming_GPUkey(JCTC.W_time, device_id, iteration)] = W_times[device_id]
        jc_timing.timings[JCTiming_GPUkey(JCTC.J_time, device_id, iteration)] = J_times[device_id]
        jc_timing.timings[JCTiming_GPUkey(JCTC.K_time, device_id, iteration)] = K_times[device_id]
        jc_timing.timings[JCTiming_GPUkey(JCTC.GPU_H_add_time, device_id, iteration)] = H_times[device_id]
        jc_timing.timings[JCTiming_GPUkey(JCTC.GPU_density_time, device_id, iteration)] = density_times[device_id]
        jc_timing.timings[JCTiming_GPUkey(JCTC.gpu_fock_time, device_id, iteration)] = gpu_fock_times[device_id]
        jc_timing.timings[JCTiming_GPUkey(JCTC.GPU_non_zero_coeff_time, device_id, iteration)] = non_zero_coeff_times[device_id]
    end
    jc_timing.timings[JCTiming_key(JCTC.fock_gpu_cpu_copy_time, iteration)] = fock_copy_time
    jc_timing.timings[JCTiming_key(JCTC.total_fock_gpu_time, iteration)] = total_fock_gpu_time


end


#todo to remove branching I need a map from screened[1d index] to unscreened 2d[p,q] indices 
function form_screened_density_kernel!(screened_density::CuDeviceArray{Float64}, density::CuDeviceArray{Float64}, 
    sparse_pq_index_map::CuDeviceArray{Int64}, p::Int64)
    
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    for pp in index:stride:p
        for qq in 1:pp-1
            if sparse_pq_index_map[pp, qq] == 0
                continue
            else 
                @inbounds screened_density[sparse_pq_index_map[pp, qq]] = 2.0*density[pp, qq] # symmetric multiplication 2.0* for off diagonal
            end
        end
        @inbounds screened_density[sparse_pq_index_map[pp, pp]] = density[pp, pp]  # and 1.0* for diagonal
    end  
end

function form_screened_density!(scf_data::SCFData, device_id::Int64)
    p = scf_data.μ
    density = scf_data.gpu_data.device_density[device_id]

    screened_density = scf_data.gpu_data.device_screened_density[device_id]
    occupied_orbital_coefficients = scf_data.gpu_data.device_occupied_orbital_coefficients[device_id]

    CUBLAS.gemm!('T', 'N', 1.0, occupied_orbital_coefficients, occupied_orbital_coefficients, 0.0, density)
    CUDA.synchronize()
    
    sparse_pq_index_map = scf_data.gpu_data.sparse_pq_index_map[device_id]

    numblocks = ceil(Int64, p/256)
    threads = min(256, p)

    @cuda threads=threads blocks=numblocks form_screened_density_kernel!(screened_density, density, sparse_pq_index_map, p)
    CUDA.synchronize()

end

# todo this could be sped up a bit 
# there is some parallelism that could be exploited here, and better yet merge this with the code in ScreenedDF.jl that forms the original ranges, 
# the slowest bits have been sped up 
# it is staying relatively constant scaling as the system size increases
# if it becomes a problem, this can be revisisted 
function setup_gpu_screening_data!(scf_data::SCFData, num_devices::Int64)
    n_ranges = 0
    p = scf_data.μ
    n_ranges_arr = zeros(Int64, p)
    Threads.@threads for pp in 1:p
        n_ranges_arr[pp] = length(scf_data.screening_data.non_zero_ranges[pp])
    end
    n_ranges = sum(n_ranges_arr)

    p_non_zero_ranges_start = zeros(Int64, p) #the nth range that corresponds to the first range for p 
    p_non_zero_ranges_start[1] = 1
    for pp in 2:p
        p_non_zero_ranges_start[pp] = p_non_zero_ranges_start[pp-1] + n_ranges_arr[pp-1]
    end
    scf_data.gpu_data.n_screened_occupied_orbital_ranges = n_ranges

    range_p = Array{Int64}(undef, n_ranges)
    range_start = Array{Int64}(undef, n_ranges)
    range_end = Array{Int64}(undef, n_ranges)
    range_sparse_start = Array{Int64}(undef, n_ranges)
    range_sparse_end = Array{Int64}(undef, n_ranges)




    range_index = 1
    Threads.@threads for pp in 1:p 
        pp_range_index = 1
        for range_index in p_non_zero_ranges_start[pp]:p_non_zero_ranges_start[pp]+n_ranges_arr[pp]-1
            range_p[range_index] = pp
            range_start[range_index] = scf_data.screening_data.non_zero_ranges[pp][pp_range_index][1]
            range_end[range_index] = scf_data.screening_data.non_zero_ranges[pp][pp_range_index][end]
            range_sparse_start[range_index] = scf_data.screening_data.non_zero_sparse_ranges[pp][pp_range_index][1]
            range_sparse_end[range_index] = scf_data.screening_data.non_zero_sparse_ranges[pp][pp_range_index][end]
            pp_range_index += 1
        end
    end
    
    Threads.@sync for device_id in 1:num_devices
        Threads.@spawn begin
            CUDA.device!(device_id-1)
            scf_data.gpu_data.device_range_p[device_id] = CUDA.zeros(Int, n_ranges)
            scf_data.gpu_data.device_range_start[device_id] = CUDA.zeros(Int, n_ranges)
            scf_data.gpu_data.device_range_end[device_id] = CUDA.zeros(Int, n_ranges)
            scf_data.gpu_data.device_range_sparse_start[device_id] = CUDA.zeros(Int, n_ranges)
            scf_data.gpu_data.device_range_sparse_end[device_id] = CUDA.zeros(Int, n_ranges)    
            scf_data.gpu_data.device_sparse_to_p[device_id] = CUDA.zeros(Int64, scf_data.screening_data.screened_indices_count)
            scf_data.gpu_data.device_sparse_to_q[device_id] = CUDA.zeros(Int64, scf_data.screening_data.screened_indices_count)
            
            scf_data.gpu_data.sparse_pq_index_map[device_id] = CUDA.zeros(Int, (p, p))

            CUDA.copyto!(scf_data.gpu_data.sparse_pq_index_map[device_id], scf_data.screening_data.sparse_pq_index_map)
            CUDA.copyto!(scf_data.gpu_data.device_range_p[device_id], range_p)
            CUDA.copyto!(scf_data.gpu_data.device_range_start[device_id], range_start)
            CUDA.copyto!(scf_data.gpu_data.device_range_end[device_id], range_end)
            CUDA.copyto!(scf_data.gpu_data.device_range_sparse_start[device_id], range_sparse_start)
            CUDA.copyto!(scf_data.gpu_data.device_range_sparse_end[device_id], range_sparse_end)
            CUDA.synchronize() 


        end
    end

    Threads.@sync for device_id in 1:num_devices
        Threads.@spawn begin
            CUDA.device!(device_id-1)
            numblocks = ceil(Int64, n_ranges/256)
            threads = min(256, n_ranges)

            @cuda threads=threads blocks=numblocks create_sparse_to_p_q_kernel(scf_data.gpu_data.device_sparse_to_p[device_id],
                scf_data.gpu_data.device_sparse_to_q[device_id], 
                scf_data.gpu_data.sparse_pq_index_map[device_id], p)
            CUDA.synchronize()
        end 
    end
end

function create_sparse_to_p_q_kernel(sparse_to_p::CuDeviceArray{Int64}, 
    sparse_to_q::CuDeviceArray{Int64}, 
    sparse_pq_index_map::CuDeviceArray{Int64},  p::Int64)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    for pp in index:stride:p
        for qq in 1:p
            sparse_index = sparse_pq_index_map[pp, qq]
            if sparse_index != 0
                sparse_to_p[sparse_index] = pp
                sparse_to_q[sparse_index] = qq
            end
        end
    end

end

#todo make this a kernel and copy from ranges that are stored on the device to avoid CPU threading 
function form_nozero_coefficient_matrix!(scf_data::SCFData, device_id :: Int64)
    CUDA.device!(device_id-1)
    

    n_ranges = scf_data.gpu_data.n_screened_occupied_orbital_ranges

    numblocks = ceil(Int64, n_ranges/256)
    threads = min(256, n_ranges)

    @cuda threads=threads blocks=numblocks build_non_zero_coefficients_kernel(scf_data.gpu_data.device_non_zero_coefficients[device_id], 
        scf_data.gpu_data.device_occupied_orbital_coefficients[device_id], 
        scf_data.gpu_data.device_range_p[device_id],
        scf_data.gpu_data.device_range_start[device_id],
        scf_data.gpu_data.device_range_end[device_id],
        scf_data.gpu_data.device_range_sparse_start[device_id],
        scf_data.gpu_data.device_range_sparse_end[device_id],
        n_ranges)
    CUDA.synchronize()
end

function build_non_zero_coefficients_kernel(non_zero_coefficients::CuDeviceArray{Float64}, 
     occupied_orbital_coefficients::CuDeviceArray{Float64},
        device_range_p::CuDeviceArray{Int64},
        device_range_start::CuDeviceArray{Int64},
        device_range_end::CuDeviceArray{Int64},
        device_range_sparse_start::CuDeviceArray{Int64},
        device_range_sparse_end::CuDeviceArray{Int64},
        n_ranges::Int64)

    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:n_ranges
        pp = device_range_p[i]
        range_start = device_range_start[i]
        range_end = device_range_end[i]
        range_sparse_start = device_range_sparse_start[i]
        # range_sparse_end = device_range_sparse_end[i]

        for j in 0:range_end-range_start
            non_zero_coefficients[:, range_sparse_start+j, pp] .= view(occupied_orbital_coefficients, :, range_start+j)
        end
    end
end

function copy_screened_J_to_fock(fock::CuDeviceArray{Float64}, J::CuDeviceArray{Float64},
        device_sparse_to_p::CuDeviceArray{Int64}, device_sparse_to_q::CuDeviceArray{Int64}, n_sparse_indicies::Int64)
    
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    
    for i = index:stride:n_sparse_indicies
        pp = device_sparse_to_p[i]
        qq = device_sparse_to_q[i]
        @inbounds fock[pp, qq] +=J[i]
    end
    return
end


function copy_screened_J_to_fock_upper_triangle(fock::CuDeviceArray{Float64}, J::CuDeviceArray{Float64},
    device_sparse_to_p::CuDeviceArray{Int64}, device_sparse_to_q::CuDeviceArray{Int64}, n_sparse_indicies::Int64)

    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x


    for i = index:stride:n_sparse_indicies
        qq = device_sparse_to_p[i]
        pp = device_sparse_to_q[i]
        @inbounds fock[pp, qq] += J[i]
    end
    return
end

function copy_lower_to_upper_kernel(A ::CuDeviceArray{Float64})
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    
    for i = index:stride:size(A, 1)
        for j = 1:size(A, 2)
            @inbounds A[j, i] = Float64(i > j) * A[i, j] + Float64(i < j)* A[j, i]  + Float64(i == j) * A[i, j] 
            # 1st term if (i,j) is in the lower triangle copy to (j,i) in the upper triangle else do nothing
            # 2nd term if (i,j) is in the upper triangle keep the value the same otherwise do nothing
            # 3rd term if on the diagonal keep the value the same else do nothing
        end
    end
   

    return
end

function copy_upper_to_lower_kernel(A ::CuDeviceArray{Float64})
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    
    for i = index:stride:size(A, 1)
        for j = i:size(A, 2)
            @inbounds A[j, i] = A[i, j] 
        end
    end
   

    return
end


function calculate_V_screened_GPU(V::CuArray, B::CuArray, density::CuArray)
    CUBLAS.gemv!('N', 1.0, B, density, 0.0, V)
    CUDA.synchronize()
end

function calculate_J_screened_GPU(J::CuArray,  B::CuArray, V::CuArray)
    CUBLAS.gemv!('T', 2.0, B, V, 0.0, J)
    CUDA.synchronize()
end

#calculate using lower trinagle only 
function calculate_J_screened_symmetric_GPU(device_density::CuArray, 
    host_J::Array{Float64,1}, J::CuArray, V::CuArray, B::CuArray, scf_data::SCFData, device_id::Int64, n_streams::Int64)

    p = scf_data.μ
    # sparse_pq_index_map = scf_data.screening_data.sparse_pq_index_map
    sparse_p_start_indices = scf_data.screening_data.sparse_p_start_indices
    CUDA.fill!(V, 0.0)
    CUDA.synchronize()

    last_index = scf_data.screening_data.screened_indices_count
    

    # for pp in 1:(p-1)
    #     range_start = sparse_pq_index_map[pp, pp]
    #     range_end = sparse_p_start_indices[pp+1] -1
    #     CUBLAS.gemv!('N', 1.0, view(B, :, range_start:range_end), view(device_density,  range_start:range_end), 1.0, V) 
    #     CUDA.synchronize()
    # end
    # CUBLAS.gemv!('N', 1.0, view(B, :, last_index:last_index),  view(device_density, last_index:last_index), 1.0, V)
    
    Threads.@sync begin
        for stream in 1:n_streams
            Threads.@spawn begin
                stream_V = scf_data.gpu_data.device_stream_coulmob_intermediate[device_id][stream]
                CUDA.fill!(stream_V, 0.0)
                CUDA.synchronize()
                for pp in stream:n_streams:p-1
                    range_start = sparse_p_start_indices[pp]
                    range_end = sparse_p_start_indices[pp+1]-1
                    CUBLAS.gemv!('N', 1.0, view(B, :, range_start:range_end), view(device_density,  range_start:range_end), 1.0, stream_V) 
                    CUDA.synchronize()
                end
            end
        end 
        Threads.@spawn begin
            CUBLAS.gemv!('N', 1.0, view(B, :, last_index:last_index),  view(device_density, last_index:last_index), 1.0, V)
            CUDA.synchronize()
        end
    end
    CUDA.synchronize()
    for stream in 1:n_streams
        stream_V = scf_data.gpu_data.device_stream_coulmob_intermediate[device_id][stream]
        CUDA.axpy!(1.0, stream_V, V)
        CUDA.synchronize()
    end


    CUDA.fill!(J, 0.0)
    CUDA.synchronize()

    Threads.@sync begin
        for stream in 1:n_streams
            Threads.@spawn begin
                stream_J = scf_data.gpu_data.device_stream_coulmob[device_id][stream]
                CUDA.fill!(stream_J, 0.0)
                CUDA.synchronize()

                for pp in stream:n_streams:p-1
                    range_start = sparse_p_start_indices[pp]
                    range_end = sparse_p_start_indices[pp+1]-1
                    CUBLAS.gemv!('T', 2.0, view(B, :, range_start:range_end), V, 1.0, view(stream_J, range_start:range_end))
                    CUDA.synchronize()
                end
            end
        end
        Threads.@spawn begin
            CUBLAS.gemv!('T', 2.0, view(B, :, last_index:last_index), V, 1.0, view(J, last_index:last_index))
            CUDA.synchronize()
        end
    end        

    for stream in 1:n_streams
        stream_J = scf_data.gpu_data.device_stream_coulmob[device_id][stream]
        CUDA.axpy!(1.0, stream_J, J)
        CUDA.synchronize()
    end

    # for pp in 1:(p-1) 
    #     range_start = sparse_p_start_indices[pp]
    #     range_end = sparse_p_start_indices[pp+1]-1
    #     CUBLAS.gemv!('T', 2.0, view(B, :, range_start:range_end), V, 1.0, view(J, range_start:range_end))
    #     CUDA.synchronize()
    # end
    # CUBLAS.gemv!('T', 2.0, view(B, :, last_index:last_index), V, 1.0, view(J, last_index:last_index))
    # CUDA.synchronize()
end

function calculate_W_screened_GPU(device_id, scf_data::SCFData, num_threads ::Int64)
    p = scf_data.μ
   
    alpha = 1.0
    beta = 0.0

    B = scf_data.gpu_data.device_B[device_id] # B intermediate from integral contraction 
    W = scf_data.gpu_data.device_exchange_intermediate[device_id] # W intermediate for exchange calculation

    non_zero_coefficients = scf_data.gpu_data.device_non_zero_coefficients[device_id]
    num_streams = min(p, num_threads)
    Threads.@sync begin 
        for pp_start in 1:num_streams
            Threads.@spawn begin
                CUDA.device!(device_id-1)
                for pp in pp_start:num_streams:p
                    K = scf_data.screening_data.non_screened_p_indices_count[pp]
                    A_cu = view(B, :, scf_data.screening_data.sparse_p_start_indices[pp]:
                        scf_data.screening_data.sparse_p_start_indices[pp]+K-1)
                    B_cu = view(non_zero_coefficients, :,1:K,pp)
                    C_cu = view(W, :,:,pp)
                    CUBLAS.gemm!('N','T', alpha, A_cu, B_cu, beta, C_cu)
                end
            end
        end 
    end
    CUDA.synchronize()
end

function calcululate_K_no_sym_GPU!(fock::CuArray{Float64,2}, W::CuArray{Float64,3},p::Int64, n_ooc::Int64, Q::Int64, device_id::Int64)
    CUBLAS.gemm!('T', 'N', -1.0, reshape(W, (Q*n_ooc, p)), reshape(W, (Q*n_ooc, p)), 0.0, fock)
    CUDA.synchronize()
end

function calculate_K_upper_diagonal_rectangle_blocks(fock::CuArray{Float64,2}, W::CuArray{Float64,3}, Q_length::Int, 
    device_id, scf_data::SCFData, scf_options::SCFOptions, num_threads::Int64)
    CUDA.device!(device_id-1)

    n_occ = scf_data.occ
    p = scf_data.μ
    Q = Q_length #device Q length

    transA = 'T'
    transB = 'N'
    alpha = -1.0
    beta = 0.0

    n_blocks = 2

    if p > 1600
        n_blocks = 16
    elseif p > 800
        n_blocks = 8
    elseif p > 400
        n_blocks = 4
    end

    if haskey(ENV, "JC_K_RECT_N_BLOCKS")
        n_blocks = parse(Int64, ENV["JC_K_RECT_N_BLOCKS"])
    end



    block_width = p÷n_blocks

    scf_data.gpu_data.device_K_block = Array{Array{CuArray{Float64}}}(undef, 1)
    scf_data.gpu_data.device_K_block[device_id] = Array{CuArray{Float64}}(undef, n_blocks)

    for block_index in 1:n_blocks
        scf_data.gpu_data.device_K_block[device_id][block_index] = CUDA.zeros(Float64, (block_index*block_width, block_width)) #todo move to iteration 1
    end
    device_K_blocks = scf_data.gpu_data.device_K_block[device_id]
    
    num_streams = min(num_threads, n_blocks)

    Threads.@sync begin
        for stream in 1:num_streams
            Threads.@spawn begin
                CUDA.device!(device_id-1)
                for block_index in stream:num_streams:n_blocks
                    M = block_index*block_width
                    block_p_range = 1:M
                    block_q_range = (block_index-1)*block_width+1:block_index*block_width


                    A = reshape(view(W, :,:, block_p_range), (Q*n_occ, M))
                    B = reshape(view(W, :,:, block_q_range), (Q*n_occ, block_width))
                    
                    CUBLAS.gemm!(transA, transB, alpha, A, B, beta, device_K_blocks[block_index]) #transpose(W[:,1:M])*W[:,q_start:q_end] = retangular block of size M X block_width 
                    # fock[block_p_range, block_q_range] .= scf_data.gpu_data.device_K_block[block_index]
                    CUDA.synchronize()

                    CUDA.copyto!(view(fock, block_p_range, block_q_range), device_K_blocks[block_index])
                    # CUDA.copyto!(view(fock, block_q_range, block_p_range), transpose(device_K_blocks[block_index]))
                    CUDA.synchronize()
                end
            end 
        end
        if p%n_blocks != 0
            Threads.@spawn begin
                M = p
                N = p%n_blocks
                K = Q*n_occ

                q_non_square_range = p-N+1:p


                A_non_square = reshape(view(W, :,:, 1:p), (K, p))
                B_non_square = reshape(view(W, :,:, q_non_square_range), (K, N))
                C_non_square = view(fock, :, q_non_square_range)

                CUBLAS.gemm!(transA, transB, alpha, A_non_square, B_non_square, beta, C_non_square)
            
                CUDA.synchronize()
                # CUDA.copyto!(view(fock, q_non_square_range, :), C_non_square)
            end
        end      
    end
end


function calculate_K_lower_diagonal_block_no_screen_GPU(host_fock::Array{Float64,2},
     fock::CuArray{Float64,2}, W::CuArray{Float64,3}, Q_length::Int, 
     device_id, scf_data::SCFData, scf_options::SCFOptions,
     lower_triangle_length :: Int64, num_threads::Int64)

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



    num_streams = min(num_threads, lower_triangle_length) #todo pass in as a parameter

    device_K_block = scf_data.gpu_data.device_K_block[device_id]

    Threads.@sync for stream_index in 1:num_streams #next step cuda streams to make use of more parallelism on the GPU device
        Threads.@async begin
            CUDA.device!(device_id-1)

            exchange_block = view(device_K_block, :,:, stream_index)

            for index in stream_index:num_streams:lower_triangle_length
                
                pp, qq = scf_data.screening_data.exchange_batch_indexes[index]
                p_range = (pp-1)*K_block_width+1:pp*K_block_width        
                q_range = (qq-1)*K_block_width+1:qq*K_block_width
        
                A = reshape(view(W, :,:, p_range), (K, K_block_width))
                B = reshape(view(W, :,:, q_range), (K, K_block_width))
        
                CUBLAS.gemm!(transA, transB, alpha, A, B, beta, exchange_block)
                CUDA.synchronize()
                CUDA.copyto!(view(fock, p_range, q_range), exchange_block)
                CUDA.synchronize()
            end
        end
    end

    if p % scf_options.df_exchange_block_width != 0 # if square blocks don't cover the entire pq space
        col_non_square_range = 1:p    
        #non square part that didn't fit in blocks
        row_non_square_range = p-(p%scf_options.df_exchange_block_width)+1:p
        
        M = length(row_non_square_range)
        N = p
        
        A_non_square = reshape(view(W, :,:, row_non_square_range), (K, M))
        B_non_square = reshape(view(W, :,:, col_non_square_range), (K, N))
        C_non_square = scf_data.gpu_data.device_non_square_K_block[device_id]#todo make this its own data in scf_data?
        
    
        CUBLAS.gemm!(transA, transB, alpha, A_non_square, B_non_square, beta, C_non_square) #W^T[M, Q*n_ooc] * W[Q*n_ooc, N] = C_non_square[M, N]
        CUDA.synchronize()

        CUDA.copyto!(view(fock, row_non_square_range,:), C_non_square)  #non contiguous memory access on the GPU bad, should use the other triangle side
        CUDA.synchronize()

    end 
end

function calculate_B_GPU!(two_center_integrals, three_center_integrals, scf_data, num_devices, num_devices_global, basis_sets, jc_timing)
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

            J_AB_time = @elapsed begin
                if setup_device_id == 1 && rank == 0
                    CUDA.copyto!(device_J_AB_invt[setup_device_id], two_center_integrals)
                    CUDA.synchronize()
                    CUSOLVER.potrf!('L', device_J_AB_invt[setup_device_id])
                    CUDA.synchronize()
                    CUSOLVER.trtri!('L', 'N', device_J_AB_invt[setup_device_id])
                    CUDA.synchronize()

                    if num_devices > 1 || n_ranks > 1
                        CUDA.copyto!(two_center_integrals, device_J_AB_invt[1]) # copy back because taking subarrays on the GPU is slow / doesn't work. Need to look into if this is possible with CUDA.jl
                    end
                end
            end
            #todo calculate the three center integrals per device (probably could directly copy to the device while it is being calculated)
            device_three_center_integrals[setup_device_id] = CUDA.zeros(Float64, size(three_center_integrals[setup_device_id]))
            CUDA.copyto!(device_three_center_integrals[setup_device_id], three_center_integrals[setup_device_id])

            device_B[setup_device_id] = CUDA.zeros(Float64, (device_Q_range_lengths[global_device_id], pq))
            device_B_send_buffers[setup_device_id] = CUDA.zeros(Float64, (max_device_Q_range_length * pq))
            host_B_send_buffers[setup_device_id] = zeros(Float64, (max_device_Q_range_length * pq))
            CUDA.synchronize()
        end #spawn
    end

    jc_timing[JCTC.form_JAB_inv_time] = J_AB_time
 
    if MPI.Comm_size(COMM) > 1
        #broadcast two_center_integrals to all ranks
        MPI.Bcast!(two_center_integrals, 0, COMM)
    end

    if n_ranks == 1 && num_devices == 1
        B_time = @elapsed begin
            CUDA.copyto!(device_J_AB_invt[1], two_center_integrals)    
            CUDA.synchronize()
            CUBLAS.trmm!('L', 'L', 'N', 'N', 1.0, device_J_AB_invt[1], device_three_center_integrals[1], device_B[1])   
            CUDA.synchronize() 
            CUDA.unsafe_free!(device_J_AB_invt[1])
            CUDA.unsafe_free!(device_three_center_integrals[1])
        end
        jc_timing[JCTC.B_time] = B_time
        return
    end

    
    B_time = @elapsed begin
        for global_recieve_device_id in 1:num_devices_global
            rec_device_Q_range_length = device_Q_range_lengths[global_recieve_device_id]
            recieve_rank = (global_recieve_device_id-1) ÷ num_devices
            rank_recieve_device_id = ((global_recieve_device_id-1) % num_devices) + 1 # one indexed device id for the rank 
            array_size = rec_device_Q_range_length*pq
            Threads.@sync for r_send_device_id in 1:num_devices
                Threads.@spawn begin
                    CUDA.device!(r_send_device_id-1) 
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
                        CUDA.synchronize()

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
                            CUDA.synchronize()
                        end
                    end
                        
                    if rank == recieve_rank # add the sent buffer to the device B matrix
                        CUDA.device!(rank_recieve_device_id-1) do 
                            device_B_send_view = reshape(view(device_B_send_buffers[rank_recieve_device_id], 1:array_size), (rec_device_Q_range_length, pq))
                            CUDA.axpy!(1.0, device_B_send_view, device_B[rank_recieve_device_id])
                            CUDA.synchronize()
                        end                
                    end
                end
            end
        end
    end

    jc_timing[JCTC.B_time] = B_time

    for device_id in 1:num_devices
        CUDA.device!(device_id-1)
        CUDA.unsafe_free!(device_B_send_buffers[device_id])
        CUDA.unsafe_free!(device_J_AB_invt[device_id])
        CUDA.unsafe_free!(device_three_center_integrals[device_id])
    end

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
# uses the indicies as they are calculated by the static_load_rank_indicies_3_eri function
# which is used for the 3-eri load balancing to ranks and devices
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



