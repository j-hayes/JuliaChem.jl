using CUDA 

#to remove branching I need a map from screened[1d index] to unscreened 2d[p,q] indices 
#not a huge performance hit at the moment so not proritiezed 
function form_screened_density_kernel_cuda!(screened_density::CuDeviceArray{Float64}, density::CuDeviceArray{Float64}, 
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

function form_screened_density_GPU!(gpu_type::CUDA_GPU, screened_density::CuArray{Float64}, density::CuArray{Float64}, sparse_pq_index_map::CuArray{Int64}, p::Int64)
    numblocks = ceil(Int64, p/256)
    threads = min(256, p)

    @cuda threads=threads blocks=numblocks form_screened_density_kernel_cuda!(screened_density, density, sparse_pq_index_map, p)
    CUDA.synchronize()
end


function create_sparse_to_p_q_kernel_cuda(sparse_to_p ::CuDeviceArray{Int64}, 
    sparse_to_q::CuDeviceArray{Int64}, 
    sparse_pq_index_map::CuDeviceArray{Int64},  
    p::Int64)
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

function create_sparse_to_p_q_GPU!(gpu_type::CUDA_GPU, device_sparse_to_p::CuArray{Int64}, device_sparse_to_q::CuArray{Int64},
    sparse_pq_index_map::CuArray{Int64}, p::Int64, n_screened_occupied_orbital_ranges::Int64)

    n_ranges = n_screened_occupied_orbital_ranges
    numblocks = ceil(Int64, n_ranges/256)
    threads = min(256, n_ranges)
  
    @cuda threads=threads blocks=numblocks create_sparse_to_p_q_kernel_cuda(device_sparse_to_p,
        device_sparse_to_q, 
        sparse_pq_index_map, p)
    GPU_synchronize(gpu_type)
end


function build_non_zero_coefficients_kernel_cuda(non_zero_coefficients::CuDeviceArray{Float64}, 
    occupied_orbital_coefficients::CuDeviceArray{Float64},
       device_range_p::CuDeviceArray{Int64},
       device_range_start::CuDeviceArray{Int64},
       device_range_end::CuDeviceArray{Int64},
       device_range_sparse_start::CuDeviceArray{Int64},
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

function form_nozero_coefficient_matrix_GPU!(gpu_type::CUDA_GPU, device_non_zero_coefficients::CuArray{Float64},
    device_occupied_orbital_coefficients::CuArray{Float64}, device_range_p::CuArray{Int64},
    device_range_start::CuArray{Int64}, device_range_end::CuArray{Int64}, device_range_sparse_start::CuArray{Int64},
    n_ranges::Int64)
    numblocks = ceil(Int64, n_ranges/256)
    threads = min(256, n_ranges)

    @cuda threads=threads blocks=numblocks build_non_zero_coefficients_kernel_cuda(device_non_zero_coefficients, 
        device_occupied_orbital_coefficients, 
        device_range_p,
        device_range_start,
        device_range_end,
        device_range_sparse_start,
        n_ranges)

    GPU_synchronize(gpu_type)

end



function copy_screened_J_to_fock_upper_triangle_cuda(fock::CuDeviceArray{Float64}, J::CuDeviceArray{Float64},
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

function copy_screened_J_to_fock_GPU!(gpu_type::CUDA_GPU, fock::CuArray{Float64}, J::CuArray{Float64}, 
    device_sparse_to_p::CuArray{Int64}, device_sparse_to_q::CuArray{Int64}, screened_indices_count::Int64) 

    numblocks = ceil(Int64, screened_indices_count/256)
    threads = min(256, screened_indices_count)
    @cuda threads=threads blocks=numblocks copy_screened_J_to_fock_upper_triangle_cuda(fock, J, device_sparse_to_p, 
        device_sparse_to_q, screened_indices_count)
                   
end


function copy_upper_to_lower_kernel_cuda(A ::CuDeviceArray{Float64})
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:size(A, 1)
        for j = axes(A, 2)
            @inbounds A[j, i] = A[i, j] 
        end
    end
    return
end

function copy_upper_to_lower_GPU!(gpu_type::CUDA_GPU, fock::CuArray{Float64}, p::Int64, screened_indices_count::Int64)
   numblocks = ceil(Int64, screened_indices_count/256)
   threads = min(256, screened_indices_count)
   @cuda threads=threads blocks=numblocks copy_upper_to_lower_kernel_cuda(fock)    
   GPU_synchronize(gpu_type)
end

export form_screened_density_GPU!, create_sparse_to_p_q_GPU!, copy_screened_J_to_fock_GPU!, copy_upper_to_lower_GPU!

