# using CUDA
# using CUDA.CUBLAS
# using CUDA.CUSOLVER
# using LinearAlgebra
# using Base.Threads
# using HDF5

# function df_rhf_fock_build_dense_GPU!(scf_data, jeri_engine_thread_df::Vector{T}, jeri_engine_thread::Vector{T2},
#     basis_sets::CalculationBasisSets,
#     occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine,T2<:RHFTEIEngine}
#     comm = MPI.COMM_WORLD
#     pq = scf_data.μ^2

#     p = scf_data.μ
#     n_ooc = scf_data.occ

#     devices = CUDA.devices()
#     num_devices = length(devices)
#     scf_data.gpu_data.number_of_devices_used = num_devices

#     if iteration == 1
#         two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
#         three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options)


#         calculate_B_dense_GPU(two_center_integrals, three_center_integrals, scf_data, num_devices)

#         #clear the memory 
#         two_center_integrals = nothing
#         three_center_integrals = nothing

#         scf_data.gpu_data.device_fock = Array{CuArray{Float64}}(undef, num_devices)
#         scf_data.gpu_data.device_coulomb_intermediate = Array{CuArray{Float64}}(undef, num_devices)
#         scf_data.gpu_data.device_exchange_intermediate = Array{CuArray{Float64}}(undef, num_devices)
#         scf_data.gpu_data.device_occupied_orbital_coefficients = Array{CuArray{Float64}}(undef, num_devices)
#         scf_data.gpu_data.device_density = Array{CuArray{Float64}}(undef, num_devices)
#         scf_data.gpu_data.host_fock = Array{Array{Float64,2}}(undef, num_devices)
#         for device_id in 1:num_devices
#             device!(device_id - 1)
#             Q = scf_data.gpu_data.device_Q_range_lengths[device_id]
#             scf_data.gpu_data.device_fock[device_id] = CUDA.zeros(Float64, (scf_data.μ, scf_data.μ))
#             scf_data.gpu_data.device_coulomb_intermediate[device_id] = CUDA.zeros(Float64, (Q))
#             scf_data.gpu_data.device_exchange_intermediate[device_id] =
#                 CUDA.zeros(Float64, (n_ooc, Q, p))
#             scf_data.gpu_data.device_occupied_orbital_coefficients[device_id] = CUDA.zeros(Float64, (scf_data.μ, scf_data.occ))
#             scf_data.gpu_data.device_density[device_id] = CUDA.zeros(Float64, (scf_data.μ, scf_data.μ))
#             scf_data.gpu_data.host_fock[device_id] = zeros(Float64, scf_data.μ, scf_data.μ)            
#         end
#     end

#     Threads.@sync for device_id in 1:num_devices
#         Threads.@spawn begin
#             device!(device_id - 1)
#             ooc = scf_data.gpu_data.device_occupied_orbital_coefficients[device_id]
#             density = scf_data.gpu_data.device_density[device_id]
            
#             B = scf_data.gpu_data.device_B[device_id]
#             Q_length = scf_data.gpu_data.device_Q_range_lengths[device_id]
#             V = scf_data.gpu_data.device_coulomb_intermediate[device_id]
#             W = scf_data.gpu_data.device_exchange_intermediate[device_id]
#             fock = scf_data.gpu_data.device_fock[device_id]
#             CUDA.copyto!(ooc, occupied_orbital_coefficients)

#             CUBLAS.gemm!('N', 'T', 1.0, ooc, ooc, 0.0, density)
#             CUBLAS.gemv!('N', 1.0, reshape(B, (Q_length, pq)), reshape(density, pq), 0.0, V)
#             CUBLAS.gemv!('T', 2.0, reshape(B, (Q_length, pq)), V, 0.0, reshape(fock, pq))
#             CUBLAS.gemm!('T', 'T', 1.0, ooc, reshape(B, (Q_length * p, p)), 0.0, reshape(W, (n_ooc, Q_length* p)))
#             CUBLAS.gemm!('T', 'N', -1.0, reshape(W, (n_ooc * Q_length, p)), reshape(W, (n_ooc * Q_length, p)), 1.0, fock)
#             CUDA.copyto!(scf_data.gpu_data.host_fock[device_id], fock) 
#             CUDA.synchronize()          
#         end
#     end


#     scf_data.two_electron_fock .= scf_data.gpu_data.host_fock[1]
#     for device_id in 2:num_devices
#         axpy!(1.0, scf_data.gpu_data.host_fock[device_id], scf_data.two_electron_fock)
#     end
# end

# function calculate_B_dense_GPU(two_center_integrals, three_center_integrals, scf_data, num_devices)
#     pq = scf_data.μ^2


#     device_J_AB_invt = Array{CuArray{Float64}}(undef, num_devices)
#     scf_data.gpu_data.device_B = Array{CuArray{Float64}}(undef, num_devices)
#     device_B = scf_data.gpu_data.device_B
#     device_three_center_integrals = Array{CuArray{Float64}}(undef, num_devices)
#     scf_data.gpu_data.device_B_send_buffers = Array{CuArray{Float64}}(undef, num_devices)
#     host_B_send_buffers = Array{Array{Float64,1}}(undef, num_devices)


#     device_B_send_buffers = scf_data.gpu_data.device_B_send_buffers
#     indices_per_device,
#     device_Q_range_starts,
#     device_Q_range_ends,
#     device_Q_indices,
#     device_Q_range_lengths,
#     max_device_Q_range_length = calculate_device_ranges_dense(scf_data, num_devices)


#     scf_data.gpu_data.device_Q_range_lengths = device_Q_range_lengths
#     scf_data.gpu_data.device_Q_range_starts = device_Q_range_starts
#     scf_data.gpu_data.device_Q_range_ends = device_Q_range_ends
#     scf_data.gpu_data.device_Q_indices = device_Q_indices
    
#     indicies_per_device = device_Q_range_lengths
#     two_eri_GPU = device_J_AB_invt

#     println("size of three center integrals: $(size(three_center_integrals))")

#     three_center_integrals_indicies = LinearIndices(size(three_center_integrals))

#     Threads.@sync for device_id in 1:num_devices
#         Threads.@spawn begin
#             device!(device_id - 1)
#             # buffer for J_AB_invt for each device max size needed is A*A 
#             # for certain B calculations the device will only need a subset of this
#             # and will reference it with a view referencing the front of the underlying array
#             device_J_AB_invt[device_id] = CUDA.zeros(Float64, (scf_data.A, scf_data.A))
#             device_three_center_integrals[device_id] = CUDA.zeros(Float64, (scf_data.μ*scf_data.μ, device_Q_range_lengths[device_id]))

#             CUDA.copyto!(device_J_AB_invt[device_id], two_center_integrals)

#             device_pointer = pointer(device_three_center_integrals[device_id])
#             host_pointer = pointer(three_center_integrals, three_center_integrals_indicies[1,1,device_Q_range_starts[device_id]])

#             CUDA.unsafe_copyto!(device_pointer, host_pointer, device_Q_range_lengths[device_id]*pq)
           
#             device_B[device_id] = CUDA.zeros(Float64, (device_Q_range_lengths[device_id], scf_data.μ, scf_data.μ))
#             device_B_send_buffers[device_id] = CUDA.zeros(Float64, (max_device_Q_range_length * scf_data.μ * scf_data.μ))      
#             host_B_send_buffers[device_id] = zeros(Float64, (max_device_Q_range_length * scf_data.μ * scf_data.μ))      
     
#             scf_data.gpu_data.device_Q_range_lengths[device_id] = device_Q_range_lengths[device_id]
#             CUDA.synchronize()          

#         end
#     end

#     LAPACK.potrf!('L', two_center_integrals)
#     LAPACK.trtri!('L', 'N', two_center_integrals)


#     Threads.@sync for device_id in 1:num_devices
#         Threads.@spawn begin
#             CUDA.copyto!(device_J_AB_invt[device_id], two_center_integrals)#use the same J_AB_invt for all devices to avoid inconsistencies from CUSOLVER    
#         end
#     end

#     #renames from script 
#     indicies_per_device = device_Q_range_lengths
#     two_eri_GPU = device_J_AB_invt
#     two_eri_host = two_center_integrals
#     three_center_integrals_GPU = device_three_center_integrals
#     B_GPU = device_B
#     B_send_buffer = device_B_send_buffers
#     B_send_buffer_host = host_B_send_buffers
#     device_start_indicies = device_Q_range_starts
#     device_end_indices = device_Q_range_ends

#     for recv_device_id in 1:num_devices
#         recieve_device_num_aux_indicies = indicies_per_device[recv_device_id]
#         Threads.@threads for send_device_id in 1:num_devices
#             send_device_num_aux_indicies = indicies_per_device[send_device_id]
#             CUDA.device!(send_device_id-1)
#             two_eri_view = reshape(
#                 view(two_eri_GPU[send_device_id], 1:recieve_device_num_aux_indicies*send_device_num_aux_indicies),
#                 recieve_device_num_aux_indicies, send_device_num_aux_indicies)

#            CUDA.copyto!(two_eri_view, two_eri_host[                     
#                 device_start_indicies[recv_device_id]:device_end_indices[recv_device_id],
#                 device_start_indicies[send_device_id]:device_end_indices[send_device_id]])

          
#             if send_device_id == recv_device_id
#                 CUBLAS.gemm!('N', 'T', 1.0,  
#                 two_eri_view,
#                 three_center_integrals_GPU[send_device_id],
#                 0.0, 
#                 reshape(B_GPU[recv_device_id], (recieve_device_num_aux_indicies, pq)))
#             else
#                 B_send_buffer_device_view = reshape(
#                     view(B_send_buffer[send_device_id], 1:recieve_device_num_aux_indicies*pq), 
#                     (recieve_device_num_aux_indicies, pq))
#                 CUBLAS.gemm!('N', 'T', 1.0,  two_eri_view, 
#                 three_center_integrals_GPU[send_device_id],
#                      0.0, B_send_buffer_device_view)

#                 CUDA.copyto!(B_send_buffer_host[send_device_id], 1, B_send_buffer[send_device_id], 1, recieve_device_num_aux_indicies*pq)
#             end
#             CUDA.synchronize()          
#         end
        
#         for send_device_id in 1:num_devices
        
#             if send_device_id == recv_device_id
#                 continue
#             end

#             CUDA.device!(recv_device_id-1)
#             CUDA.copyto!(B_send_buffer[recv_device_id], 1, B_send_buffer_host[send_device_id], 1, recieve_device_num_aux_indicies*pq)
#             send_buffer_view = view(B_send_buffer[recv_device_id], 1:recieve_device_num_aux_indicies*pq)
#             CUDA.axpy!(1.0, send_buffer_view, B_GPU[recv_device_id])
#         end       
#     end

# end

# function calculate_device_ranges_dense(scf_data, num_devices)
#     indices_per_device = scf_data.A ÷ num_devices
#     device_Q_range_starts = []
#     device_Q_range_ends = []
#     device_Q_range_lengths = []
#     println("calculate_device_ranges_dense num_devices: $num_devices ")

#     for device_id in 1:num_devices
#         push!(device_Q_range_starts, (device_id - 1) * indices_per_device + 1)
#         push!(device_Q_range_ends, device_id * indices_per_device)
#     end

#     device_Q_range_ends[end] = scf_data.A

#     # device_Q_range_starts = 1:indices_per_device+1:scf_data.A
#     # device_Q_range_ends = device_Q_range_starts .+ indices_per_device
    
#     device_Q_indices = [device_Q_range_starts[i]:device_Q_range_ends[i] for i in 1:num_devices]
#     device_Q_indices[end] = device_Q_range_starts[end]:scf_data.A
    
#     device_Q_range_lengths = length.(device_Q_indices)
    
#     max_device_Q_range_length = maximum(device_Q_range_lengths)
#     return indices_per_device, device_Q_range_starts, device_Q_range_ends, device_Q_indices, device_Q_range_lengths, max_device_Q_range_length
# end
