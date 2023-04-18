using MPI 
using Random
using LinearAlgebra
function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    time = @elapsed do_some_blas() 
    if MPI.Comm_rank(comm) == 0
        println("done in $(time) seconds")
    end
    MPI.Finalize()
end   

function do_some_blas()
    comm = MPI.COMM_WORLD
    μ = 600
    ν = μ
    A = 1500
    two_center = rand(Float64, (A,A))
    two_center = two_center*two_center'
    two_center = two_center + 5*I

    three_center = rand(Float64, (μ*ν, A))
    D = Array{Float64,2}(rand(Float64, (μ *ν, A)))
    D_permuted = Array{Float64,3}(zeros(Float64, (μ,ν, A)))
    calculate_D!(D,D_permuted,two_center, three_center, μ, A)

    MPI.Barrier(comm)
end


function calculate_D!(D,D_permuted,two_center_integrals, three_center_integrals, μ, A)
    comm = MPI.COMM_WORLD
    # this needs to be mpi parallelized

    time_1 = time() 
    J_AB_invt = convert(Array{Float64}, transpose(cholesky(Hermitian(two_center_integrals, :L)).L \I))
    MPI.Barrier(comm)
    if MPI.Comm_rank(comm) == 0
        println("done in J_AB_invt: $(time() - time_1) seconds")
    end

    time_1 = time()
    ## todo get this from a parameter to make this more readable 
    μμ = μ
    νν = μ
    AA = A
    # use transpose transpose gemm transpose to perform tensor contraction 
    # Linv_T is already a 2D matrix so no need to reshape, and is in correct order
    D = reshape(D, (μμ*νν,AA))
    three_center_integrals =  reshape(three_center_integrals, (μμ*νν,AA))
    BLAS.gemm!('N', 'N', 1.0, three_center_integrals, J_AB_invt, 0.0, D)
    D = reshape(D, (μμ, νν, AA))
    D_permuted = reshape(D_permuted, (μμ,  AA,νν))
    permutedims!(D_permuted, D, (1,3,2))
    D_permuted = reshape(D_permuted, (νν*AA, μμ))
    MPI.Barrier(comm)
    if MPI.Comm_rank(comm) == 0
        println("done in the rest: $(time() - time_1) seconds")
    end
  end # end function calculate_D


main()