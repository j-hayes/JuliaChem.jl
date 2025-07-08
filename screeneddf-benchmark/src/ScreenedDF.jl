# filepath: /screeneddf-benchmark/screeneddf-benchmark/src/ScreenedDF.jl
function calculate_K_lower_diagonal_block_no_screen(W::AbstractMatrix{Float64}, μ::Int, occ::Int, K_block_width::Int, Q::Int)
    transA = true
    transB = false
    alpha = -1.0
    beta = 0.0
    linear_indices = LinearIndices(W)

    M = K_block_width
    N = K_block_width
    K = Q * occ

    exchange_blocks = zeros(Float64, K_block_width, K_block_width)

    for index in 1:μ
        pp = index
        qq = index

        p_range = (pp - 1) * K_block_width + 1:pp * K_block_width
        q_range = (qq - 1) * K_block_width + 1:qq * K_block_width

        A_ptr = pointer(W, linear_indices[1, 1, p_range.start])
        B_ptr = pointer(W, linear_indices[1, 1, q_range.start])
        C_ptr = pointer(exchange_blocks, 1)

        call_gemm!(Val(transA), Val(transB), M, N, K, alpha, A_ptr, B_ptr, beta, C_ptr)

        # Assuming two_electron_fock is a preallocated matrix
        two_electron_fock = zeros(Float64, μ, μ)
        two_electron_fock[p_range, q_range] .= exchange_blocks
        if pp != qq
            two_electron_fock[q_range, p_range] .= transpose(exchange_blocks)
        end
    end
end

function call_gemm!(transA::Val, transB::Val,
    M::Int, N::Int, K::Int,
    alpha::Float64, A::Ptr{Float64}, B::Ptr{Float64},
    beta::Float64, C::Ptr{Float64})

    convtrans(V::Val{false}) = 'N'
    convtrans(V::Val{true}) = 'T'

    lda = transA == Val(false) ? M : K
    ldb = transB == Val(false) ? K : N
    ldc = M

    ccall((:dgemm_64_, BLAS.libblas), Nothing,
        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
            Ref{BlasInt}, Ref{Float64}, Ptr{Float64}, Ref{BlasInt},
            Ptr{Float64}, Ref{BlasInt}, Ref{Float64}, Ptr{Float64},
            Ref{BlasInt}),
        convtrans(transA), convtrans(transB), M, N, K,
        alpha, A, lda, B, ldb, beta, C, ldc)
end