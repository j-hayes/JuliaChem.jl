using CUDA
mutable struct ScreeningData
    sparse_pq_index_map
    basis_function_screen_matrix
    sparse_p_start_indices
    non_screened_p_indices_count
    #non_zero_coefficients belongs in scf_data not screening todo move
    non_zero_coefficients # GTFOCK paper equation 4 "In order to use optimized matrix multiplication library functions to compute W(i, Q), for each p, we need a dense matrix consisting of the nonzero rows of C(r, i)."
    screened_triangular_indicies
    shell_screen_matrix
    screened_triangular_indicies_to_2d
    r_ranges #todo rename
    B_ranges #todo rename
    block_screen_matrix::Array{Bool,2}
    blocks_to_calculate::Array{Int,1}
    exchange_batch_indexes::Array{Tuple{Int,Int}}
    non_zero_ranges::Array{Array{UnitRange{Int}}}
    non_zero_sparse_ranges::Array{Array{UnitRange{Int}}}
    triangular_indices_count::Int
    screened_indices_count::Int
    K_block_width::Int
end

mutable struct SCFGPUData
    device_Q_range_lengths::Array{Int,1}
    device_Q_range_starts::Array{Int,1}
    device_Q_range_ends::Array{Int,1}
    device_Q_indices::Array{UnitRange{Int},1}
    device_B::Array{Union{Nothing,CuArray{Float64}},1}
    device_B_send_buffers::Array{Union{Nothing,CuArray{Float64}},1}
    device_fock::Array{Union{Nothing,CuArray{Float64}},1}
    device_exchange_intermediate::Array{Union{Nothing,CuArray{Float64}},1}
    device_occupied_orbital_coefficients::Array{Union{Nothing,CuArray{Float64}},1}
    device_coulomb::Array{Union{Nothing,CuArray{Float64}},1}
    device_stream_coulmob::Array{Union{Nothing,Array{CuArray{Float64}}},1}
    device_coulomb_intermediate::Array{Union{Nothing,CuArray{Float64}},1}
    device_stream_coulmob_intermediate::Array{Union{Nothing,Array{CuArray{Float64}}},1} #todo this data could be shared between stream V and J 
    device_density::Array{Union{Nothing,CuArray{Float64}},1}
    device_screened_density::Array{Union{Nothing,CuArray{Float64}},1}
    device_non_zero_coefficients::Array{Union{Nothing,CuArray{Float64}},1}
    device_K_block#::Array{Union{Nothing,CuArray{Float64}},1}
    device_non_square_K_block::Array{Union{Nothing,CuArray{Float64}},1}
    host_coulomb::Array{Array{Float64,1},1}
    host_fock::Array{Array{Float64,2},1}
    device_H::CuArray{Float64} #only copied to rank 0 GPU 1 because it only needs to be added to one of the partial fock matricies 
    #metadata for screening 
    sparse_pq_index_map::Array{CuArray{Int64,2},1}
    device_range_p::Array{CuArray{Int64,1},1}
    device_range_start::Array{CuArray{Int64,1},1}
    device_range_end::Array{CuArray{Int64,1},1}
    device_range_sparse_start::Array{CuArray{Int64,1},1}
    device_range_sparse_end::Array{CuArray{Int64,1},1}
    device_sparse_to_p::Array{CuArray{Int64,1},1}
    device_sparse_to_q::Array{CuArray{Int64,1},1}
    n_screened_occupied_orbital_ranges::Int64
    number_of_devices_used::Int64
end

mutable struct SCFData
    D
    D_tilde
    two_electron_fock
    two_electron_fock_GPU
    thread_two_electron_fock
    coulomb_intermediate
    density
    occupied_orbital_coefficients
    density_array
    J
    K
    k_blocks
    screening_data::ScreeningData
    gpu_data::SCFGPUData
    μ::Int
    occ::Int
    A::Int
    scf_iteration::Int
end

function initialize!(gpu_data::SCFGPUData, num_devices::Int64)
    gpu_data.device_fock = Array{CuArray{Float64}}(undef, num_devices)
    gpu_data.device_coulomb_intermediate = Array{CuArray{Float64}}(undef, num_devices)
    gpu_data.device_coulomb = Array{CuArray{Float64}}(undef, num_devices)
    gpu_data.device_stream_coulmob = Array{Array{CuArray{Float64}}}(undef, num_devices)
    gpu_data.device_stream_coulmob_intermediate = Array{Array{CuArray{Float64}}}(undef, num_devices)


    gpu_data.device_exchange_intermediate = Array{CuArray{Float64}}(undef, num_devices)
    gpu_data.device_occupied_orbital_coefficients = Array{CuArray{Float64}}(undef, num_devices)
    gpu_data.device_density = Array{CuArray{Float64}}(undef, num_devices)
    gpu_data.device_screened_density = Array{CuArray{Float64}}(undef, num_devices)
    gpu_data.device_non_zero_coefficients = Array{Array{CuArray{Float64}}}(undef, num_devices)
    gpu_data.device_K_block = Array{CuArray{Float64}}(undef, num_devices)
    gpu_data.device_non_square_K_block = Array{CuArray{Float64}}(undef, num_devices)
    gpu_data.host_coulomb = Array{Array{Float64,1}}(undef, num_devices)

    gpu_data.device_range_p = Array{CuArray{Int64,1}}(undef, num_devices)
    gpu_data.device_range_start = Array{CuArray{Int64,1}}(undef, num_devices)
    gpu_data.device_range_end = Array{CuArray{Int64,1}}(undef, num_devices)
    gpu_data.device_range_sparse_start = Array{CuArray{Int64,1}}(undef, num_devices)
    gpu_data.device_range_sparse_end = Array{CuArray{Int64,1}}(undef, num_devices)
    gpu_data.device_sparse_to_p = Array{CuArray{Int64,1}}(undef, num_devices)
    gpu_data.device_sparse_to_q = Array{CuArray{Int64,1}}(undef, num_devices)

    gpu_data.sparse_pq_index_map = Array{CuArray{Int64,2}}(undef, num_devices)

    gpu_data.host_fock = Array{Array{Float64,2}}(undef, num_devices)

end

function SCFData()
    sd = ScreeningData([], [], [], [], [], [], [], [], [], [], falses(1, 1), zeros(Int, 0), Array{Tuple{Int,Int}}(undef, 0),
        Array{Array{UnitRange{Int}}}(undef, 0), Array{Array{UnitRange{Int}}}(undef, 0), 0, 0, 0)
    gpu_data = SCFGPUData([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], CuArray{Float64}(undef, 0), [], 0, 0)
    return SCFData([], [], [], [], [], [], [], [], [], [], [], [], sd, gpu_data, 0, 0, 0, 0)
end

function clear_gpu_data(scf_data::SCFData)
    # number_of_gpus = scf_data.gpu_data.number_of_devices_used
    # gpu_data::SCFGPUData = scf_data.gpu_data

    # for device_id in 1:number_of_gpus
    #     CUDA.device!(device_id - 1)
    #     if !isnothing(gpu_data.device_fock)
    #         CUDA.unsafe_free!(gpu_data.device_fock[device_id])
    #     end
    #     if !isnothing(gpu_data.device_B)
    #         CUDA.unsafe_free!(gpu_data.device_B[device_id])
    #     end
    #     if !isnothing(gpu_data.device_B_send_buffers) && length(gpu_data.device_B_send_buffers) >= device_id
    #         CUDA.unsafe_free!(gpu_data.device_B_send_buffers[device_id])
    #     end
    #     if !isnothing(gpu_data.device_fock)
    #         CUDA.unsafe_free!(gpu_data.device_fock[device_id])
    #     end
    #     if !isnothing(gpu_data.device_exchange_intermediate)
    #         CUDA.unsafe_free!(gpu_data.device_exchange_intermediate[device_id])
    #     end
    #     if !isnothing(gpu_data.device_occupied_orbital_coefficients)
    #         CUDA.unsafe_free!(gpu_data.device_occupied_orbital_coefficients[device_id])
    #     end
    #     if !isnothing(gpu_data.device_coulomb)
    #         CUDA.unsafe_free!(gpu_data.device_coulomb[device_id])
    #     end
    #     if !isnothing(gpu_data.device_stream_coulmob)
    #         for stream in gpu_data.device_stream_coulmob[device_id]
    #             CUDA.unsafe_free!(stream)
    #         end
    #     end
    #     if !isnothing(gpu_data.device_coulomb_intermediate)
    #         CUDA.unsafe_free!(gpu_data.device_coulomb_intermediate[device_id])
    #     end
    #     if !isnothing(gpu_data.device_stream_coulmob_intermediate)
    #         for stream in gpu_data.device_stream_coulmob_intermediate[device_id]
    #             CUDA.unsafe_free!(stream)
    #         end
    #     end
    #     if !isnothing(gpu_data.device_density)
    #         CUDA.unsafe_free!(gpu_data.device_density[device_id])
    #     end
    #     if !isnothing(gpu_data.device_screened_density)
    #         CUDA.unsafe_free!(gpu_data.device_screened_density[device_id])
    #     end
    #     if !isnothing(gpu_data.device_non_zero_coefficients) 
    #         CUDA.unsafe_free!(gpu_data.device_non_zero_coefficients[device_id])
    #     end
    #     if !isnothing(gpu_data.device_K_block)
    #         CUDA.unsafe_free!(gpu_data.device_K_block[device_id])
    #     end
    #     if !isnothing(gpu_data.device_non_square_K_block)
    #         CUDA.unsafe_free!(gpu_data.device_non_square_K_block[device_id])
    #     end
    #     if !isnothing(gpu_data.device_H)
    #         CUDA.unsafe_free!(gpu_data.device_H)
    #     end
    #     if !isnothing(gpu_data.sparse_pq_index_map)
    #         CUDA.unsafe_free!(gpu_data.sparse_pq_index_map[device_id])
    #     end
    #     if !isnothing(gpu_data.device_range_p)
    #         CUDA.unsafe_free!(gpu_data.device_range_p[device_id])
    #     end
    #     if !isnothing(gpu_data.device_range_start)
    #         CUDA.unsafe_free!(gpu_data.device_range_start[device_id])
    #     end
    #     if !isnothing(gpu_data.device_range_end)
    #         CUDA.unsafe_free!(gpu_data.device_range_end[device_id])
    #     end
    #     if !isnothing(gpu_data.device_range_sparse_start)
    #         CUDA.unsafe_free!(gpu_data.device_range_sparse_start[device_id])
    #     end
    #     if !isnothing(gpu_data.device_range_sparse_end)
    #         CUDA.unsafe_free!(gpu_data.device_range_sparse_end[device_id])
    #     end
    #     if !isnothing(gpu_data.device_sparse_to_p)
    #         CUDA.unsafe_free!(gpu_data.device_sparse_to_p[device_id])
    #     end
    #     if !isnothing(gpu_data.device_sparse_to_q)
    #         CUDA.unsafe_free!(gpu_data.device_sparse_to_q[device_id])
    #     end
    # end
    # GC.gc(true)
    # for device_id in 1:number_of_gpus
    #     CUDA.device!(device_id - 1)
    #     CUDA.reclaim()
    # end
end

export SCFData, initialize!, SCFGPUData, ScreeningData, clear_gpu_data