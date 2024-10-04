using CUDA
using CUDA.CUBLAS
using CUDA.CUSOLVER
using LinearAlgebra
using Base.Threads
using HDF5

function df_rhf_fock_build_dense_GPU!(scf_data, jeri_engine_thread_df::Vector{T}, jeri_engine_thread::Vector{T2},
    basis_sets::CalculationBasisSets,
    occupied_orbital_coefficients, iteration, scf_options::SCFOptions, host_H) where {T<:DFRHFTEIEngine,T2<:RHFTEIEngine}
    comm = MPI.COMM_WORLD
    pq = scf_data.μ^2
    rank = MPI.Comm_rank(comm)

    if rank == 0 && MPI.Comm_size(comm) > 1
        println("Dense GPU algorithm is not supported for MPI runs with more than 1 rank")
        MPI.Abort(comm, 1)
        exit()
    elseif rank != 0
        MPI.Abort(comm, 1)
        exit()
    end
    device_id = 1
    device!(0)


    p = scf_data.μ
    n_ooc = scf_data.occ
    Q = scf_data.A

    num_devices = 1 
    scf_data.gpu_data.number_of_devices_used = num_devices

    if iteration == 1
        two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
        three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options)


        calculate_B_dense_GPU(two_center_integrals, three_center_integrals, scf_data, num_devices)

        #clear the memory 
        two_center_integrals = nothing
        three_center_integrals = nothing

        scf_data.gpu_data.device_fock = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_coulomb_intermediate = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_exchange_intermediate = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_occupied_orbital_coefficients = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_density = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.host_fock = Array{Array{Float64,2}}(undef, num_devices)
        scf_data.gpu_data.device_fock[device_id] = CUDA.zeros(Float64, (scf_data.μ, scf_data.μ))
        scf_data.gpu_data.device_coulomb_intermediate[device_id] = CUDA.zeros(Float64, (Q))
        scf_data.gpu_data.device_exchange_intermediate[device_id] =
            CUDA.zeros(Float64, (n_ooc, Q, p))
        scf_data.gpu_data.device_occupied_orbital_coefficients[device_id] = CUDA.zeros(Float64, (scf_data.μ, scf_data.occ))
        scf_data.gpu_data.device_density[device_id] = CUDA.zeros(Float64, (scf_data.μ, scf_data.μ))
        scf_data.gpu_data.host_fock[device_id] = zeros(Float64, scf_data.μ, scf_data.μ)    
        scf_data.gpu_data.device_H = CUDA.zeros(Float64, (scf_data.μ, scf_data.μ))           

        CUDA.synchronize()        
        CUDA.copyto!(scf_data.gpu_data.device_H, host_H)
        CUDA.synchronize()
    end

    ooc = scf_data.gpu_data.device_occupied_orbital_coefficients[device_id]
    density = scf_data.gpu_data.device_density[device_id]
    
    B = scf_data.gpu_data.device_B[device_id]
    V = scf_data.gpu_data.device_coulomb_intermediate[device_id]
    W = scf_data.gpu_data.device_exchange_intermediate[device_id]
    fock = scf_data.gpu_data.device_fock[device_id]
    H = scf_data.gpu_data.device_H
    

            
    total_fock_gpu_time = @elapsed begin
        
        CUDA.copyto!(ooc, occupied_orbital_coefficients)
        CUDA.synchronize()

        CUBLAS.gemm!('N', 'T', 1.0, ooc, ooc, 0.0, density)
        CUDA.synchronize()          
        
        J_time = @elapsed begin
            CUBLAS.gemv!('N', 1.0, reshape(B, (Q, pq)), reshape(density, pq), 0.0, V)
            CUDA.synchronize()          
            CUBLAS.gemv!('T', 2.0, reshape(B, (Q, pq)), V, 0.0, reshape(fock, pq))
            CUDA.synchronize()          
        end
        W_time = @elapsed begin
            CUBLAS.gemm!('T', 'T', 1.0, ooc, reshape(B, (Q * p, p)), 0.0, reshape(W, (n_ooc, Q* p)))
            CUDA.synchronize()  
        end
        K_time = @elapsed begin
            CUBLAS.gemm!('T', 'N', -1.0, reshape(W, (n_ooc * Q, p)), reshape(W, (n_ooc * Q, p)), 1.0, fock)
            CUDA.copyto!(scf_data.gpu_data.host_fock[device_id], fock) 
            CUDA.synchronize()          
        end
        H_time = @elapsed begin
            CUDA.axpy!(1.0, H, fock)
            CUDA.synchronize()   
        end
    end     
    total_time = total_fock_gpu_time 
    println("gpu timings: $(total_fock_gpu_time) W_time: $(W_time) J_time: $(J_time) K_time: $(K_time) H_time: $(H_time)")
    println("total time: $(total_time)")

    CUDA.copyto!(scf_data.gpu_data.host_fock[1], scf_data.gpu_data.device_fock[device_id])

    scf_data.two_electron_fock .= scf_data.gpu_data.host_fock[1]
    for device_id in 2:num_devices
        axpy!(1.0, scf_data.gpu_data.host_fock[device_id], scf_data.two_electron_fock)
    end
end

function calculate_B_dense_GPU(two_center_integrals, three_center_integrals, scf_data, num_devices)


    scf_data.gpu_data.device_B = Array{CuArray{Float64}}(undef, num_devices)
    device_B = scf_data.gpu_data.device_B
    device_three_center_integrals = Array{CuArray{Float64}}(undef, num_devices)
    scf_data.gpu_data.device_B_send_buffers = Array{CuArray{Float64}}(undef, num_devices)

    device_J_AB_invt = CUDA.zeros(Float64, (scf_data.A, scf_data.A))

    CUDA.device!(0)
    CUDA.copyto!(device_J_AB_invt, two_center_integrals)
    CUDA.synchronize()

    CUSOLVER.potrf!('L', device_J_AB_invt)
    CUDA.synchronize()
    CUSOLVER.trtri!('L', 'N',  device_J_AB_invt)
    CUDA.synchronize()

    CUDA.device!(0)


    pq = scf_data.μ^2

    three_center_integrals = permutedims(three_center_integrals, [3,1,2]) #todo this shouldn't be necessary it should be formed in this shape for 1 rank 1 device run
    device_three_center_integrals = CUDA.zeros(Float64, (scf_data.A, pq))
    device_B[1] = CUDA.zeros(Float64, (scf_data.A, pq))
    CUDA.synchronize()
    CUDA.copyto!(device_three_center_integrals, three_center_integrals)
    CUDA.synchronize()

    CUBLAS.trmm!('L', 'L', 'N', 'N', 1.0,  device_J_AB_invt, device_three_center_integrals, device_B[1])   
    CUDA.synchronize()

end

function calculate_device_ranges_dense(scf_data, num_devices)
    indices_per_device = scf_data.A ÷ num_devices
    device_Q_range_starts = []
    device_Q_range_ends = []
    device_Q_range_lengths = []
    println("calculate_device_ranges_dense num_devices: $num_devices ")

    for device_id in 1:num_devices
        push!(device_Q_range_starts, (device_id - 1) * indices_per_device + 1)
        push!(device_Q_range_ends, device_id * indices_per_device)
    end

    device_Q_range_ends[end] = scf_data.A

    # device_Q_range_starts = 1:indices_per_device+1:scf_data.A
    # device_Q_range_ends = device_Q_range_starts .+ indices_per_device
    
    device_Q_indices = [device_Q_range_starts[i]:device_Q_range_ends[i] for i in 1:num_devices]
    device_Q_indices[end] = device_Q_range_starts[end]:scf_data.A
    
    device_Q_range_lengths = length.(device_Q_indices)
    
    max_device_Q_range_length = maximum(device_Q_range_lengths)
    return indices_per_device, device_Q_range_starts, device_Q_range_ends, device_Q_indices, device_Q_range_lengths, max_device_Q_range_length
end
