using CUDA
mutable struct ScreeningData
    sparse_pq_index_map
    basis_function_screen_matrix
    shifted_basis_function_screen_matrix
    coefficient_shifts :: Array{Array{Tuple{Int,Int},1},1}
    sparse_p_start_indices
    non_screened_p_indices_count
    #non_zero_coefficients belongs in scf_data not screening todo move
    non_zero_coefficients # GTFOCK paper equation 4 "In order to use optimized matrix multiplication library functions to compute W(i, Q), for each p, we need a dense matrix consisting of the nonzero rows of C(r, i)."
    screened_triangular_indicies
    shell_screen_matrix
    screened_triangular_indicies_to_2d
    r_ranges #todo rename
    B_ranges #todo rename
    block_screen_matrix :: Array{Bool,2}
    blocks_to_calculate :: Array{Int,1}
    exchange_batch_indexes :: Array{Tuple{Int, Int}}
    triangular_indices_count :: Int
    screened_indices_count :: Int
    K_block_width :: Int
end

mutable struct SCFGPUData
    device_Q_range_lengths :: Array{Int,1}
    device_Q_range_starts :: Array{Int,1}
    device_Q_range_ends :: Array{Int,1}
    device_Q_indices :: Array{UnitRange{Int},1}
    device_B :: Array{Union{Nothing, CuArray{Float64}},1}
    device_B_send_buffers :: Array{Union{Nothing, CuArray{Float64}},1}
    device_fock :: Array{Union{Nothing, CuArray{Float64}},1}
    device_coulomb_intermediate :: Array{Union{Nothing, CuArray{Float64}},1}
    device_exchange_intermediate :: Array{Union{Nothing, CuArray{Float64}},1}
    device_occupied_orbital_coefficients :: Array{Union{Nothing, CuArray{Float64}},1}
    device_coulomb :: Array{Union{Nothing, CuArray{Float64}},1}
    device_density :: Array{Union{Nothing, CuArray{Float64}},1}
    device_non_zero_coefficients :: Array{Union{Nothing, CuArray{Float64}},1}
    device_K_block :: Array{Union{Nothing, CuArray{Float64}},1}
    host_coulomb ::  Array{Array{Float64,1},1}
    host_fock ::  Array{Array{Float64,2},1}
    number_of_devices_used :: Int
    device_start_index
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
    screening_data :: ScreeningData
    gpu_data :: SCFGPUData
    μ :: Int
    occ :: Int
    A :: Int
    scf_iteration :: Int
end


function SCFData()
    sd = ScreeningData([],[],[], [], [], [], [], [], [], [], [],
         [], falses(1,1), zeros(Int,0), Array{Tuple{Int, Int}}(undef,0), 0, 0, 0)
    gpu_data = SCFGPUData([], [], [], [], [], [], [], [], [], [], [], [], [], [], [],[], 0, 0)
    return SCFData([], [], [],[], [], [], [], [],[] ,[],[],[], sd, gpu_data, 0, 0 ,0,0)
end

export SCFData