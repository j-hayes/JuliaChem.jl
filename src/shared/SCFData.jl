using CUDA
mutable struct ScreeningData
    sparse_pq_index_map
    basis_function_screen_matrix
    sparse_p_start_indices
    non_screened_p_indices_count
    screened_triangular_indicies
    shell_screen_matrix
    screened_triangular_indicies_to_2d
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
    device_stream_coulmob_intermediate::Array{Union{Nothing,Array{CuArray{Float64}}},1} #todo this data could be shared between stream V and J #todo remove if J sym algorithm is not implemented for GPU
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
    device_Q_index_lengths::Array{Int,1}
end

mutable struct SCFData
    D
    D_tilde
    two_electron_fock
    coulomb_intermediate
    density
    occupied_orbital_coefficients
    non_zero_coefficients # GTFOCK paper equation 4 "In order to use optimized matrix multiplication library functions to compute W(i, Q), for each p, we need a dense matrix consisting of the nonzero rows of C(r, i)."
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
    sd = ScreeningData([], [], [], [], [], [], [], falses(1, 1), zeros(Int, 0), Array{Tuple{Int,Int}}(undef, 0),
        Array{Array{UnitRange{Int}}}(undef, 0), Array{Array{UnitRange{Int}}}(undef, 0), 0, 0, 0)
    gpu_data = SCFGPUData([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], CuArray{Float64}(undef, 0), [], 0, 0, [])
    return SCFData([], [], [], [], [],[], [], [], [], [], [], sd, gpu_data, 0, 0, 0, 0)
end

export SCFData, initialize!, SCFGPUData, ScreeningData, clear_gpu_data