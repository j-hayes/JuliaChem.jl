using CUDA

struct CUDA_GPU <: GPU_Type end
const CUDAAF64 = CUDA.CuArray{Float64}
const CUDAAI64 = CUDA.CuArray{Int64}


function get_default_gpu_data_cuda(num_devices) :: SCFGPUData_generic
    gpu_data = SCFGPUData_generic{CuArray{Float64}, CuArray{Int64}}(
        [], [], [], [], [], 
        [], [], [], [], [], 
        [], [], [], [], [],
        [], [], [], [] ,[],
        [], [], [], [], [],
        [], 0, 0, [], CUDA_GPU())
        initialize_generic!(CUDAAF64, CUDAAI64, gpu_data, num_devices, CUDA_GPU())
        GPU_synchronize(gpu_data.GPU_Type)
    return gpu_data
end

function get_array_types(::CUDA_GPU)
    return CUDAAF64, CUDAAI64
end

function CUDA_GPU_enabled()
    return CUDA.functional()
end

function set_gpu_device(device_id::Int64, ::CUDA_GPU)
    CUDA.device!(device_id)
end

function GPU_synchronize(::CUDA_GPU)
    CUDA.synchronize()
end 

function GPU_num_devices(gpu_type::CUDA_GPU) :: Int64
    return length(CUDA.devices())
end

function GPU_zeros(::CUDA_GPU, T::Type, dims::Any)
    return CUDA.zeros(T, dims...)
end

export get_default_gpu_data_cuda, SCFGPUData_cuda, CUDA_GPU_enabled, set_gpu_device, GPU_trtri!, CUDA_GPU, GPU_num_devices, GPU_synchronize, get_array_types, GPU_zeros