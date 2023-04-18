using LinearAlgebra
using Random

function main()

    dim_1 = 150
    dim_2 = dim_1^2

    rand_nums = rand(Float64, (dim_1*dim_1*dim_2))
    integrals = Array{Float64, 3}(undef, (dim_1, dim_1, dim_2))
    integrals_2 = Array{Float64, 2}(undef, (dim_1*dim_1,dim_2))

    integrals_2 = reshape(integrals_2, (dim_1*dim_1*dim_2))
    integrals_2 .= rand_nums
    
    rand_nums = reshape(rand_nums, (dim_1, dim_1, dim_2))
    integrals .= rand_nums

    integrals_2 = reshape(integrals_2, (dim_1*dim_1, dim_2))
    integrals = reshape(integrals, (dim_1*dim_1, dim_2))


    println(typeof(integrals_2))
    println("size of integrals: ", size(integrals))
    println("size of integrals_2: ", size(integrals_2))


    output = Array{Float64, 2}(undef, (dim_1*dim_1, dim_2))
    BLAS.gemm!('N', 'N', 1.0, integrals, integrals, 0.0, output)
    BLAS.gemm!('N', 'N', 1.0, integrals_2, integrals_2, 0.0, output)

    @time BLAS.gemm!('N', 'N', 1.0, integrals, integrals, 0.0, output)
    @time BLAS.gemm!('N', 'N', 1.0, integrals_2, integrals_2, 0.0, output)

end
println("hello np")
# main()