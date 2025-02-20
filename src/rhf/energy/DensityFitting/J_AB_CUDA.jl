using CUDA
using CUDA.CUBLAS
using CUDA.CUSOLVER

function GPU_trtri!(gpu_type::CUDA_GPU, uplo::Char, diag::Char, A::CuArray{Float64})
    CUBLAS.trtri!(uplo, diag, A)
end

function GPU_calculate_J_AB_INV!(gpu_type::CUDA_GPU, two_center_integrals::Array{Float64}, gpu_two_center_integrals::CuArray{Float64})
    copyto!(device_J_AB_invt[1], two_center_integrals)
    GPU_synchronize(gpu_type)
    CUSOLVER.potrf!('L', device_J_AB_invt[1])
    GPU_synchronize(gpu_type)
    GPU_trtri!(gpu_type, 'L', 'N',  device_J_AB_invt[1])
    GPU_synchronize(gpu_type)        
end