using CUDA
using CUDA.CUBLAS
using CUDA.CUSOLVER

function GPU_trtri!(gpu_type::CUDA_GPU, uplo::Char, diag::Char, A::CuArray{Float64})
    CUSOLVER.trtri!(uplo, diag, A)
end

function calculate_J_AB_INV_GPU!(gpu_type::CUDA_GPU, two_center_integrals::Array{Float64}, gpu_two_center_integrals::CuArray{Float64})
    copyto!(gpu_two_center_integrals, two_center_integrals)
    GPU_synchronize(gpu_type)
    CUSOLVER.potrf!('L', gpu_two_center_integrals)
    GPU_synchronize(gpu_type)
    GPU_trtri!(gpu_type, 'L', 'N',  gpu_two_center_integrals)
    GPU_synchronize(gpu_type)        
end