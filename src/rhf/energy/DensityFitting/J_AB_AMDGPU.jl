using AMDGPU
using AMDGPU.rocBLAS
using LinearAlgebra

function GPU_trtri!(gpu_type::AMD_GPU, uplo::Char, diag::Char, A::ROCArray{Float64})
    LinearAlgebra.LAPACK.chkuplo(uplo)
    n = LinearAlgebra.LAPACK.checksquare(A)
    lda = max(1, stride(A, 2))            
    devinfo = ROCVector{Cint}(undef, 1)

    AMDGPU.rocSOLVER.rocsolver_dtrtri(rocBLAS.handle(), uplo, diag, n, A, lda, devinfo)

    info = AMDGPU.@allowscalar devinfo[1]
    AMDGPU.unsafe_free!(devinfo)
    LinearAlgebra.LAPACK.chkargsok(LinearAlgebra.BlasInt(info))
end

function calculate_J_AB_INV_GPU!(gpu_type::AMD_GPU, two_center_integrals::Array{Float64}, gpu_two_center_integrals::ROCArray{Float64})
    copyto!(device_J_AB_invt[1], two_center_integrals)
    GPU_synchronize(gpu_type)
    AMDGPU.rocSOLVER.potrf!('L', device_J_AB_invt[1])
    GPU_synchronize(gpu_type)
    GPU_trtri!(gpu_type, 'L', 'N',  device_J_AB_invt[1])
    GPU_synchronize(gpu_type)        
end