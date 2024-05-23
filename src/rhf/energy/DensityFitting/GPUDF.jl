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
    # num_devices = length(devices)
    num_devices = 2
    
    if iteration == 1
        two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
        three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options)

        calculate_B_GPU!(two_center_integrals, three_center_integrals, scf_data, num_devices)
        
        #clear the memory 
        two_center_integrals = nothing
        three_center_integrals = nothing

        scf_data.gpu_data.device_fock = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_coulomb_intermediate = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_exchange_intermediate = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_occupied_orbital_coefficients =  Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_density =  Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.host_fock = Array{Array{Float64,2}}(undef, num_devices)

        for device_id in 1:num_devices
            device!(device_id-1)
            Q = scf_data.gpu_data.device_Q_range_lengths[device_id]
            scf_data.gpu_data.device_fock[device_id] = CUDA.CuArray{Float64}(undef, (scf_data.μ, scf_data.μ))
            scf_data.gpu_data.device_coulomb_intermediate[device_id] = CUDA.CuArray{Float64}(undef, (Q))
            scf_data.gpu_data.device_exchange_intermediate[device_id] = 
                CUDA.CuArray{Float64}(undef, (n_ooc, Q, p))
            scf_data.gpu_data.device_occupied_orbital_coefficients[device_id] = CUDA.CuArray{Float64}(undef, (scf_data.μ, scf_data.occ))
            scf_data.gpu_data.device_density[device_id] = CUDA.CuArray{Float64}(undef, (scf_data.μ, scf_data.μ))
            scf_data.gpu_data.host_fock[device_id] = zeros(scf_data.μ, scf_data.μ)
            fill!(scf_data.gpu_data.device_fock[device_id], 0.0)
            fill!(scf_data.gpu_data.device_coulomb_intermediate[device_id], 0.0)
            fill!(scf_data.gpu_data.device_exchange_intermediate[device_id], 0.0)
            fill!(scf_data.gpu_data.device_occupied_orbital_coefficients[device_id], 0.0)
            fill!(scf_data.gpu_data.device_density[device_id], 0.0)
        end
        
        

    end

    #Threads.@threads  
    for device_id in 1:num_devices
        device!(device_id-1)
        ooc = scf_data.gpu_data.device_occupied_orbital_coefficients[device_id]
        density = scf_data.gpu_data.device_density[device_id]
        B = scf_data.gpu_data.device_B[device_id]
        Q_length = scf_data.gpu_data.device_Q_range_lengths[device_id]
        V = scf_data.gpu_data.device_coulomb_intermediate[device_id]
        W = scf_data.gpu_data.device_exchange_intermediate[device_id]
        fock = scf_data.gpu_data.device_fock[device_id]
        copyto!(ooc, occupied_orbital_coefficients)

        CUBLAS.gemm!('N', 'T', 1.0, ooc, ooc, 0.0, density)
        CUBLAS.gemv!('N', 1.0, reshape(B, (Q_length, pq)), reshape(density, pq), 0.0, V)
        CUBLAS.gemv!('T', 2.0, reshape(B, (Q_length, pq)), V , 0.0, reshape(fock, pq))
        CUBLAS.gemm!('T', 'T' , 1.0, ooc, reshape(B, (Q_length*p,p)), 0.0, reshape(W, (n_ooc,p*Q_length)))
        CUBLAS.gemm!('T', 'N', -1.0, reshape(W, (n_ooc*Q_length, p)), reshape(W, (n_ooc*Q_length, p)), 1.0, fock)
    end
    
    #Threads.@threads  
    for device_id in 1:num_devices
        device!(device_id-1)
        copyto!(scf_data.gpu_data.host_fock[device_id], scf_data.gpu_data.device_fock[device_id])
    end
    
    scf_data.two_electron_fock .= scf_data.gpu_data.host_fock[1]
    for device_id in 2:num_devices
        axpy!(1.0, scf_data.gpu_data.host_fock[device_id], scf_data.two_electron_fock)
    end

end

function calculate_B_GPU!(two_center_integrals, three_center_integrals, scf_data, num_devices)
    pq = scf_data.μ^2


    device_J_AB_invt = Array{CuArray{Float64}}(undef, num_devices)
    scf_data.gpu_data.device_B = Array{CuArray{Float64}}(undef, num_devices)
    device_B = scf_data.gpu_data.device_B
    device_three_center_integrals =  Array{CuArray{Float64}}(undef, num_devices)
    scf_data.gpu_data.device_B_send_buffers =  Array{CuArray{Float64}}(undef, num_devices)
    device_B_send_buffers = scf_data.gpu_data.device_B_send_buffers
    indices_per_device,
     device_Q_range_starts, 
     device_Q_range_ends, 
     device_Q_indices, 
     device_Q_range_lengths, 
     max_device_Q_range_length = calculate_device_ranges(scf_data, num_devices)

    scf_data.gpu_data.device_Q_range_lengths = device_Q_range_lengths
    scf_data.gpu_data.device_Q_range_starts = device_Q_range_starts
    scf_data.gpu_data.device_Q_range_ends = device_Q_range_ends
    scf_data.gpu_data.device_Q_indices = device_Q_indices


    three_eri_linear_indices = LinearIndices(three_center_integrals)
   
    #Threads.@threads  
    for device_id in 1:num_devices
        device!(device_id-1)
        # buffer for J_AB_invt for each device max size needed is A*A 
        # for certain B calculations the device will only need a subset of this
        # and will reference it with a view referencing the front of the underlying array
        device_J_AB_invt[device_id] = CUDA.CuArray{Float64}(undef, (scf_data.A, scf_data.A)) 
        copyto!(device_J_AB_invt[device_id], two_center_integrals)

        CUSOLVER.potrf!('L', device_J_AB_invt[device_id])
        CUSOLVER.trtri!('L', 'N', device_J_AB_invt[device_id])
        device_three_center_integrals[device_id] = CUDA.CuArray{Float64}(undef, (scf_data.μ, scf_data.μ, device_Q_range_lengths[device_id]))
        
        three_center_integral_size = device_Q_range_lengths[device_id]*pq
        device_pointer = pointer(device_three_center_integrals[device_id],1)
        device_pointer_start = three_eri_linear_indices[1, 1, device_Q_range_starts[device_id]]
        host_pointer = pointer(three_center_integrals, device_pointer_start)
        unsafe_copyto!(device_pointer, host_pointer,three_center_integral_size)
        # copyto!(device_three_center_integrals[device_id], view(three_center_integrals, :,:,device_Q_indices[device_id]))

        device_B[device_id] = CUDA.CuArray{Float64}(undef, (device_Q_range_lengths[device_id], scf_data.μ, scf_data.μ))
        fill!(device_B[device_id], 0.0)
        device_B_send_buffers[device_id] = CUDA.CuArray{Float64}(undef, (max_device_Q_range_length*scf_data.μ*scf_data.μ))
        fill!(device_B_send_buffers[device_id], 0.0)
    end
    device!(0)
    copyto!(two_center_integrals, device_J_AB_invt[1]) # copy back because taking subarrays on the GPU is slow / doesn't work. Need to look into if this is possible with CUDA.jl
    J_AB_INV = two_center_integrals #simple rename to avoid confusion 
    
    for recieve_device_id in 1:num_devices 
        rec_device_Q_range_length = device_Q_range_lengths[recieve_device_id]
        
        #Threads.@threads  
        for send_device_id in 1:num_devices
            device!(send_device_id-1)    
            send_device_Q_range_length = device_Q_range_lengths[send_device_id]
            J_AB_invt_for_device = J_AB_INV[device_Q_indices[recieve_device_id], device_Q_indices[send_device_id]]

            device_J_AB_inv_count = rec_device_Q_range_length*send_device_Q_range_length # total number of elements in the J_AB_invt matrix for the device
            
            copyto!(device_J_AB_invt[send_device_id], J_AB_invt_for_device) #copy the needed J_AB_invt data to the device 
            three_eri_view = reshape(device_three_center_integrals[send_device_id], (pq, send_device_Q_range_length))
            J_AB_INV_view = reshape(view(device_J_AB_invt[send_device_id], 1:device_J_AB_inv_count), (rec_device_Q_range_length,send_device_Q_range_length))
            
            if send_device_id != recieve_device_id
                send_B_view = view(device_B_send_buffers[send_device_id], 1:rec_device_Q_range_length*pq)
                CUBLAS.gemm!('N', 'T', 1.0, J_AB_INV_view, three_eri_view, 
                    0.0, reshape(send_B_view, (rec_device_Q_range_length, pq)))
            else
                CUBLAS.gemm!('N', 'T', 1.0, J_AB_INV_view, three_eri_view, 1.0,
                    reshape(device_B[recieve_device_id], (rec_device_Q_range_length, pq)))
            end
        end
        
        device!(recieve_device_id-1)
        array_size = rec_device_Q_range_length*pq

        for send_device_id in 1:num_devices
            if send_device_id == recieve_device_id
                continue
            end               
            
            unsafe_copyto!(device_B_send_buffers[recieve_device_id],1, device_B_send_buffers[send_device_id], 1, array_size)
            CUBLAS.axpy!(array_size, 1.0, 
                reshape(view(device_B_send_buffers[recieve_device_id], 1:array_size), (array_size)),
                reshape(view(device_B[recieve_device_id], 1:array_size), (array_size)))
        end
    end
end

function calculate_device_ranges(scf_data, num_devices)
    indices_per_device = scf_data.A ÷ num_devices 
    device_Q_range_starts = 1:indices_per_device+1:scf_data.A
    device_Q_range_ends = device_Q_range_starts .+ indices_per_device

    device_Q_indices = [device_Q_range_starts[i]:device_Q_range_ends[i] for i in 1:num_devices]
    device_Q_indices[end] = device_Q_range_starts[end]:scf_data.A
    device_Q_range_lengths = length.(device_Q_indices)
    max_device_Q_range_length = maximum(device_Q_range_lengths)
    return indices_per_device, device_Q_range_starts, device_Q_range_ends, device_Q_indices, device_Q_range_lengths, max_device_Q_range_length
end