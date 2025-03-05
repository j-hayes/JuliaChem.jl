using oneAPI

function create_sparse_to_p_q_kernel_oneAPI(sparse_to_p, 
    sparse_to_q,
    sparse_pq_index_map,  
    p::Int64)

    qq = get_global_id()
    if qq > p
        return nothing
    end
    # stride = gridDim().x * blockDim().x

    for pp in 1:p
        @inbounds sparse_index = sparse_pq_index_map[pp, qq]
        if sparse_index != 0
            @inbounds sparse_to_p[sparse_index] = pp
            @inbounds sparse_to_q[sparse_index] = qq
        end
    end
    return nothing
end

function create_sparse_to_p_q_GPU!(gpu_type::oneAPI_GPU, device_sparse_to_p::oneArray{Int64}, device_sparse_to_q::oneArray{Int64},
    sparse_pq_index_map::oneArray{Int64}, p::Int64, n_screened_occupied_orbital_ranges::Int64)
    numThreads = 256
    threads = min(p , numThreads)
    blocks = ceil(Int, p / threads)

    @oneapi items=threads groups=blocks create_sparse_to_p_q_kernel_oneAPI(device_sparse_to_p,
        device_sparse_to_q, 
        sparse_pq_index_map, p)

    GPU_synchronize(gpu_type)
end

function build_non_zero_coefficients_kernel_oneAPI(non_zero_coefficients, 
    occupied_orbital_coefficients,
       device_range_p,
       device_range_start,
       device_range_end,
       device_range_sparse_start,
       n_ranges::Int64)

    i = get_global_id()

    if i > n_ranges
        return nothing
    end
 
    pp = device_range_p[i]
    range_start = device_range_start[i]
    range_end = device_range_end[i]
    range_sparse_start = device_range_sparse_start[i]
    for j in 0:range_end-range_start
        non_zero_coefficients[:, range_sparse_start+j, pp] .= view(occupied_orbital_coefficients, :, range_start+j)
    end
    return nothing

end

function form_nozero_coefficient_matrix_GPU!(gpu_type::oneAPI_GPU,
    device_non_zero_coefficients::oneArray{Float64},
    device_occupied_orbital_coefficients::oneArray{Float64},
    device_range_p::oneArray{Int64},
    device_range_start::oneArray{Int64},
    device_range_end::oneArray{Int64},
    device_range_sparse_start::oneArray{Int64},
    n_ranges::Int64) 

    numThreads = 256
    threads = min(n_ranges , numThreads)
    blocks = ceil(Int, n_ranges / threads)

    @oneapi items=threads groups=blocks build_non_zero_coefficients_kernel_oneAPI(device_non_zero_coefficients, 
        device_occupied_orbital_coefficients, 
        device_range_p,
        device_range_start,
        device_range_end,
        device_range_sparse_start,
        n_ranges)
    GPU_synchronize(gpu_type)
end


#to remove branching I need a map from screened[1d index] to unscreened 2d[p,q] indices 
#not a huge performance hit at the moment so not proritiezed 
function form_screened_density_kernel_oneAPI!(screened_density, density, sparse_pq_index_map, p::Int64)
    
    pp = get_global_id()
    if pp > p
        return nothing
    end
    for qq in 1:pp-1
        if sparse_pq_index_map[pp, qq] == 0
            continue
        else 
            @inbounds screened_density[sparse_pq_index_map[pp, qq]] = 2.0*density[pp, qq] # symmetric multiplication 2.0* for off diagonal
        end
    end
    @inbounds screened_density[sparse_pq_index_map[pp, pp]] = density[pp, pp]  # and 1.0* for diagonal
    return nothing
end

function form_screened_density_GPU!(gpu_type::oneAPI_GPU, screened_density::oneArray{Float64}, density::oneArray{Float64}, sparse_pq_index_map::oneArray{Int64}, p::Int64)
    
    numThreads = 256
    threads = min(p , numThreads)
    blocks = ceil(Int, p / threads)

    @oneapi items=threads groups=blocks form_screened_density_kernel_oneAPI!(screened_density, density, sparse_pq_index_map, p)
    GPU_synchronize(gpu_type)

end


function copy_screened_J_to_fock_upper_triangle_oneAPI(fock, J,
    device_sparse_to_p, device_sparse_to_q,  n_sparse_indicies::Int64)

    i = get_global_id()

    if i > n_sparse_indicies
        return nothing
    end

    @inbounds value = J[i]
    @inbounds qq = device_sparse_to_p[i]
    @inbounds pp = device_sparse_to_q[i]
    @inbounds fock[pp, qq] += J[i]
    return nothing 
end

function copy_screened_J_to_fock_GPU!(gpu_type::oneAPI_GPU, 
    fock::oneArray{Float64}, 
    J::oneArray{Float64},
    device_sparse_to_p::oneArray{Int64}, 
    device_sparse_to_q::oneArray{Int64}, 
    screened_indices_count::Int64)

    
    numThreads = min(256, screened_indices_count)
    blocks = ceil(Int, screened_indices_count / numThreads)
  
    @oneapi items=numThreads groups=blocks copy_screened_J_to_fock_upper_triangle_oneAPI(fock, J, device_sparse_to_p, device_sparse_to_q, screened_indices_count)
    GPU_synchronize(gpu_type)


end


function copy_upper_to_lower_kernel_oneAPI(A)
    i = get_global_id()
    if i > size(A, 1)
        return nothing
    end
    for j = 1:i
        @inbounds A[j, i] = A[i, j] 
    end
    return nothing
end

function copy_upper_to_lower_GPU!(gpu_type::oneAPI_GPU, fock::oneArray{Float64}, p::Int64, screened_indices_count::Int64)
    numThreads = 256
    threads = min(p , numThreads)
    blocks = ceil(Int, p / threads)                        

    @oneapi items=threads groups=blocks copy_upper_to_lower_kernel_oneAPI(fock)
    GPU_synchronize(gpu_type) 
end



export create_sparse_to_p_q_GPU!, form_nozero_coefficient_matrix_GPU!, form_screened_density_GPU!, copy_screened_J_to_fock_GPU!, copy_upper_to_lower_GPU!