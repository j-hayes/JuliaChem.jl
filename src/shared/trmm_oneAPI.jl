using oneAPI
using oneAPI.oneMKL

#CUBLAS/AMDGPU.rocBLAS/oneapi.onemkl have different function signatures and one does trmm in place
function trmm_GPU!(gpu_type::oneAPI_GPU, side:: Char, uplo::Char, transA::Char, diag::Char, alpha::Float64, A::oneAPI.oneArray{Float64}, B::oneAPI.oneArray{Float64}, C::oneAPI.oneArray{Float64})
    oneAPI.oneMKL.trmm!(side, uplo, transA, diag, alpha, A, B)
    GPU_synchronize(gpu_type)
    C .= B # should do a better job of setting the pointers that are used in scf_data so that 
    GPU_synchronize(gpu_type)
end

export trmm_GPU!