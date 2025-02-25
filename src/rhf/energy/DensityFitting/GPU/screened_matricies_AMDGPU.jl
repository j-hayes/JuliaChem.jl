using AMDGPU

#to remove branching I need a map from screened[1d index] to unscreened 2d[p,q] indices 
#not a huge performance hit at the moment so not proritiezed 
function form_screened_density_kernel_amd!(screened_density, density, sparse_pq_index_map, p::Int64)
    
    pp = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
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

function form_screened_density_GPU!(gpu_type::AMD_GPU, screened_density::ROCArray{Float64}, density::ROCArray{Float64}, sparse_pq_index_map::ROCArray{Int64}, p::Int64)
    #amd 
    numThreads = 512
    threads = min(p , numThreads)
    blocks = ceil(Int, p / threads)

    @roc groupsize=threads gridsize=blocks form_screened_density_kernel_amd!(screened_density, density, sparse_pq_index_map, p)
    GPU_synchronize(gpu_type)
end


#todo make this a 2d kernel? 
# inspired by https://github.com/JuliaORNL/JACC.jl/blob/main/ext/JACCAMDGPU/JACCAMDGPU.jl
function create_sparse_to_p_q_kernel_amd(sparse_to_p, 
    sparse_to_q,
    sparse_pq_index_map,  
    p::Int64)

    qq = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
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

# inspired by https://github.com/JuliaORNL/JACC.jl/blob/main/ext/JACCAMDGPU/JACCAMDGPU.jl
function create_sparse_to_p_q_GPU!(gpu_type::AMD_GPU, device_sparse_to_p::ROCArray{Int64}, device_sparse_to_q::ROCArray{Int64},
    sparse_pq_index_map::ROCArray{Int64}, p::Int64, n_screened_occupied_orbital_ranges::Int64)
    numThreads = 512
    threads = min(p , numThreads)
    blocks = ceil(Int, p / threads)

    @roc groupsize=threads gridsize=blocks create_sparse_to_p_q_kernel_amd(device_sparse_to_p,
        device_sparse_to_q, 
        sparse_pq_index_map, p)

    GPU_synchronize(gpu_type)
end


function build_non_zero_coefficients_kernel_amd(non_zero_coefficients, 
    occupied_orbital_coefficients,
       device_range_p,
       device_range_start,
       device_range_end,
       device_range_sparse_start,
       n_ranges::Int64)

   i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x

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
end


function form_nozero_coefficient_matrix_GPU!(gpu_type::AMD_GPU,
    device_non_zero_coefficients::ROCArray{Float64},
    device_occupied_orbital_coefficients::ROCArray{Float64},
    device_range_p::ROCArray{Int64},
    device_range_start::ROCArray{Int64},
    device_range_end::ROCArray{Int64},
    device_range_sparse_start::ROCArray{Int64},
    n_ranges::Int64) 

    numThreads = 512
    threads = min(n_ranges , numThreads)
    blocks = ceil(Int, n_ranges / threads)

    @roc groupsize=threads gridsize=blocks build_non_zero_coefficients_kernel_amd(device_non_zero_coefficients, 
        device_occupied_orbital_coefficients, 
        device_range_p,
        device_range_start,
        device_range_end,
        device_range_sparse_start,
        n_ranges)

    GPU_synchronize(gpu_type)

end


function copy_screened_J_to_fock_upper_triangle_amd(fock, J,
    device_sparse_to_p, device_sparse_to_q, n_sparse_indicies::Int64)

    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x

    if i > n_sparse_indicies
        return nothing
    end

    qq = device_sparse_to_p[i]
    pp = device_sparse_to_q[i]
    @inbounds fock[pp, qq] += J[i]
    return nothing 
end

function copy_screened_J_to_fock_GPU!(gpu_type::AMD_GPU, 
    fock::ROCArray{Float64}, 
    J::ROCArray{Float64},
    device_sparse_to_p::ROCArray{Int64}, 
    device_sparse_to_q::ROCArray{Int64},
    screened_indices_count::Int64)
    
    numThreads = 512
    threads = min(screened_indices_count , numThreads)
    blocks = ceil(Int, screened_indices_count / threads)

    @roc groupsize=threads gridsize=blocks copy_screened_J_to_fock_upper_triangle_amd(fock, J, device_sparse_to_p, 
        device_sparse_to_q, scf_data.screening_data.screened_indices_count)

    GPU_synchronize(gpu_type)
end

function copy_upper_to_lower_kernel_amd(A)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    if i > size(A, 1)
        return nothing
    end
    for j = axes(A, 2)
        @inbounds A[j, i] = A[i, j] 
    end
    return nothing
end

function copy_upper_to_lower_GPU!(gpu_type::AMD_GPU, fock::ROCArray{Float64}, p::Int64, screened_indices_count::Int64)
     # amd @roc
     numThreads = 512
     threads = min(p , numThreads)
     blocks = ceil(Int, p / threads)                        

     @roc groupsize=threads gridsize=blocks copy_upper_to_lower_kernel_amd(fock)
     GPU_synchronize(gpu_type) 
end

export create_sparse_to_p_q_GPU!, form_nozero_coefficient_matrix_GPU!, form_screened_density_GPU!, copy_screened_J_to_fock_GPU!, copy_upper_to_lower_GPU!