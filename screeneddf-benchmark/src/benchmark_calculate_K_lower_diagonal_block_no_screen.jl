# filepath: /screeneddf-benchmark/screeneddf-benchmark/src/benchmark_calculate_K_lower_diagonal_block_no_screen.jl
using Random
using LinearAlgebra
using BenchmarkTools

function calculate_K_lower_diagonal_block_no_screen(W::AbstractMatrix, μ::Int, occ::Int, K_block_width::Int)
    p = μ
    Q = size(W, 1)

    transA = true
    transB = false
    alpha = -1.0
    beta = 0.0
    linear_indices = LinearIndices(W)

    M = K_block_width
    N = K_block_width
    K = Q * occ

    exchange_blocks = zeros(Float64, K_block_width, K_block_width)

    for ii in 1:K_block_width
        for jj in 1:K_block_width
            A_ptr = pointer(W, linear_indices[1, 1, (ii - 1) * K_block_width + 1])
            B_ptr = pointer(W, linear_indices[1, 1, (jj - 1) * K_block_width + 1])
            C_ptr = pointer(exchange_blocks, (ii - 1) * K_block_width + jj)

            ccall((:dgemm_64_, Base.Libs.BLAS.libblas), Nothing,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                Ref{BlasInt}, Ref{Float64}, Ptr{Float64}, Ref{BlasInt},
                Ptr{Float64}, Ref{BlasInt}, Ref{Float64}, Ptr{Float64},
                Ref{BlasInt}),
                'T', 'N', M, N, K,
                alpha, A_ptr, K, B_ptr, K, beta, C_ptr, M)
        end
    end

    return exchange_blocks
end

function benchmark_calculate_K_lower_diagonal_block_no_screen()
    μ = 100
    occ = 10
    K_block_width = 10
    W = randn(μ * occ, μ * occ)

    @btime calculate_K_lower_diagonal_block_no_screen($W, $μ, $occ, $K_block_width)
end

benchmark_calculate_K_lower_diagonal_block_no_screen()