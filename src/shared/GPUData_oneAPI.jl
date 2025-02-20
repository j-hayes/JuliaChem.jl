using oneAPI
using oneAPI.oneMKL
using LinearAlgebra

const oneAF64 = oneAPI.oneArray{Float64}
const oneAI64 = oneAPI.oneArray{Int64}
struct oneAPI_GPU <: GPU_Type end


function get_default_gpu_dataoneAPI(num_devices) :: SCFGPUData_generic

    gpu_data = SCFGPUData_generic{oneArray{Float64}, oneArray{Int64}}(
        [], [], [], [], [], 
        [], [], [], [], [], 
        [], [], [], [], [],
        [], [], [], [] ,[],
        [], [], [], [], [],
        [], 0, 0, [], oneAPI_GPU())
    initialize_generic!(oneAF64, oneAI64, gpu_data, num_devices, oneAPI_GPU())
    return gpu_data
end

function oneAPI_GPU_enabled()
    return oneAPI.functional()
end

# set the device to the AMD GPU
# device_id is the device number in zero based indexign 
# GPU_Type is the type of GPU being used, parameter for aiding multiple dispatch
function set_gpu_device(device_id::Int64, gpu_type::oneAPI_GPU)
    oneAPI.device!(device_id+1)
end

function get_array_types(gpu_type::oneAPI_GPU)
    return oneAF64, oneAI64
end

function GPU_zeros(gpu_type::oneAPI_GPU ,T::Type, dims::Any)
    return oneAPI.zeros(T, dims...)
end

function GPU_synchronize(gpu_type::oneAPI_GPU)
    oneAPI.synchronize()
end


function GPU_num_devices(gpu_type::oneAPI_GPU) :: Int64
    return length(AMDGPU.devices())
end


export get_default_gpu_dataoneAPI, oneAPI_GPU_enabled, set_gpu_device, get_array_types, GPU_zeros, GPU_synchronize, GPU_trtri!, GPU_num_devices, oneAPI_GPU