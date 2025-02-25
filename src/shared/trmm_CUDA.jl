using CUDA.CUBLAS

#CUBLAS and AMDGPU.rocBLAS have different function signatures and one does trmm in place
function trmm_GPU!(gpu_type::CUDA_GPU, side:: Char, uplo::Char, transA::Char, diag::Char, alpha::Float64, A::CuDeviceArray{Float64}, B::CuDeviceArray{Float64}, C::CuDeviceArray{Float64})
    CUBLAS.trmm!(side, uplo, transA, diag, alpha, A, B, C)
    GPU_synchronize(gpu_type)
end

export trmm_GPU!
