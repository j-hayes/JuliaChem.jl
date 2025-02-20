using oneAPI
using LinearAlgebra

function GPU_calculate_J_AB_INV!(gpu_type::oneAPI_GPU, two_center_integrals::Array{Float64}, gpu_two_center_integrals::oneArray{Float64}) 
    # trtri does not yet exist on oneAPI.jl do this on CPU and copy 
    LinearAlgebra.LAPACK.potrf!('L', two_center_integrals)
    LinearAlgebra.LAPACK.trtri!('L', 'N', two_center_integrals)
    copyto!(gpu_two_center_integrals, two_center_integrals)
    GPU_synchronize(gpu_type)
end