using AMDGPU.rocBLAS

function trmm_GPU!(gpu_type::AMD_GPU, side::Char, uplo::Char, transA::Char, diag::Char, alpha::Float64, A::ROCArray{Float64}, B::ROCArray{Float64}, C::ROCArray{Float64})
    rocBLAS.trmm!(side, uplo, transA, diag, alpha, A, B, C)
    GPU_synchronize(gpu_type)
    C .= B # should do a better job of setting the pointers that are used in scf_data so that this isn't necessary
    GPU_synchronize(gpu_type)

end

export trmm_GPU!