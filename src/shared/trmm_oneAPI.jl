using oneAPI
using oneAPI.oneMKL

#CUBLAS/AMDGPU.rocBLAS/oneapi.onemkl have different function signatures and one does trmm in place
function trmm_GPU!(gpu_type::oneAPI_GPU, side:: Char, uplo::Char, transA::Char, diag::Char, alpha::Float64, A::oneAPI.oneArray{Float64}, B::oneAPI.oneArray{Float64}, C::oneAPI.oneArray{Float64})
    #copy B to the host 
    #trmm not working for large sizes of A,B,C
    oneAPI.oneMKL.gemm!(transA, 'N', alpha, A, B, 0.0, C)
    GPU_synchronize(gpu_type)

end

export trmm_GPU!