# using MPIPreferences

# MPIPreferences.use_system_binary()

using CUDA
# using HDF5
using LinearAlgebra
using Base.Threads
const BlasInt = LinearAlgebra.BlasInt

# function read_exchange_intermediate() :: Array{Float64}
#     #read the exchange intermediate from hdf5 file

#     h5open("./testoutputs/exchange_inter_s22_3.h5", "r") do file
#         exchange_intermediate = read(file, "exchange")
#         return exchange_intermediate
#     end
# end


#todo move this to a BLAS shared file
function call_gemm!(transA::Val, transB::Val,
    M::Int, N::Int, K::Int,
    alpha::Float64, A::Ptr{Float64}, B::Ptr{Float64},
    beta::Float64, C::Ptr{Float64})

    # Convert our compile-time transpose marker to a char for BLAS
    convtrans(V::Val{false}) = 'N'
    convtrans(V::Val{true}) = 'T'

    if transA == Val(false)
        lda = M
    else
        lda = K
    end
    if transB == Val(false)
        ldb = K
    else
        ldb = N
    end
    ldc = M

    ccall((:dgemm_64_, BLAS.libblas), Nothing,
        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
            Ref{BlasInt}, Ref{Float64}, Ptr{Float64}, Ref{BlasInt},
            Ptr{Float64}, Ref{BlasInt}, Ref{Float64}, Ptr{Float64},
            Ref{BlasInt}),
        convtrans(transA), convtrans(transB), M, N, K,
        alpha, A, lda, B, ldb, beta, C, ldc)
end


function get_triangle_matrix_length(n)::Int
    return n * (n + 1) ÷ 2
end

function do_exchange_build_square(exchange_intermediate::Array{Float64})::Array{Float64}

    W = permutedims(exchange_intermediate, (3, 2, 1))

    p = size(exchange_intermediate)[1]
    Q = size(exchange_intermediate)[2]
    i = size(exchange_intermediate)[3]
    occ = i

    exchange = zeros(Float64, p, p)
    df_exchange_block_width = 10
    K_block_width = p ÷ df_exchange_block_width

    block_lower_triangle_count = get_triangle_matrix_length(df_exchange_block_width)


    M = K_block_width
    N = K_block_width
    K = Q * occ
    transA = true
    transB = false
    alpha = -1.0
    beta = 0.0

    exchange = zeros(Float64, p, p)
    exchange_batch_indexes = Array{Tuple{Int,Int}}(undef, block_lower_triangle_count)
    the_batch_index = 1
    for iii in 1:df_exchange_block_width
        for jjj in 1:iii
            exchange_batch_indexes[the_batch_index] = (iii, jjj)
            the_batch_index += 1
        end
    end

    total_non_screened_indices = 0
    block_index = 1
    block_screen_matrix = zeros(Bool, K_block_width, K_block_width)

    n_threads = 1


    exchange_blocks = zeros(Float64, K_block_width, K_block_width, n_threads)
    linear_indices = LinearIndices(W)
    K_linear_indices = LinearIndices(exchange_blocks)


    thread = 1
    for block_index in 1:block_lower_triangle_count
        pp, qq = exchange_batch_indexes[block_index]

        p_range = (pp-1)*K_block_width+1:pp*K_block_width
        p_start = (pp - 1) * K_block_width + 1

        q_range = (qq-1)*K_block_width+1:qq*K_block_width
        q_start = (qq - 1) * K_block_width + 1

        A_ptr = pointer(W, linear_indices[1, 1, p_start])
        B_ptr = pointer(W, linear_indices[1, 1, q_start])
        C_ptr = pointer(exchange_blocks, K_linear_indices[1, 1, thread])


        call_gemm!(Val(transA), Val(transB), M, N, K, alpha, A_ptr, B_ptr, beta, C_ptr)

        exchange[p_range, q_range] .= view(exchange_blocks, :, :, thread)
        if pp != qq
            exchange[q_range, p_range] .= transpose(view(exchange_blocks, :, :, thread))
        end


    end
    println("done with ")
    return exchange
end


function main()

    #read the exchange intermediate from hdf5 file
    exchange_intermediate = read_exchange_intermediate()

    p = size(exchange_intermediate)[1]
    Q = size(exchange_intermediate)[2]
    i = size(exchange_intermediate)[3]

    println("p: $p Q: $Q i: $i")
    exchange = zeros(Float64, p, p)
    reshaped_intermediate = reshape(exchange_intermediate, (p, Q * i))

    BLAS.gemm!('N', 'T', -1.0, reshaped_intermediate, reshaped_intermediate, 0.0, exchange)



    symmetric_exchange = do_exchange_build_square(exchange_intermediate)

    println("exchange")
    display(exchange)
    println("symmetric_exchange")
    display(symmetric_exchange)
    println("max difference")

    println(maximum(exchange .- symmetric_exchange))


end

function stream_test(a,b,c)

   
    c_host = []

    CUBLAS

    # @time begin
    profile_data = CUDA.@profile trace=true  begin
        Threads.@threads for i in 1:n_batches
            CUBLAS.gemm!('N', 'N', 1.0, view(a, :, :, i), view(b, :, :, i), 0.0, view(c, :, :, i))
        end
        CUDA.device_synchronize()
        # @sync begin
        #     for i in 1:n_batches
        #         @async CUBLAS.gemm!('N', 'N', 1.0, view(a, :, :, i), view(b, :, :, i), 0.0, view(c, :, :, i))
        #     end
        # end
        # CUDA.device_synchronize()
        c_host = Array(c[:, :, 1:10])
    end
    # display(c_host[:,:,1])
    # end

    display(profile_data)
    return nothing
end

#print number of threads 
# println("Threads.nthreads() = ", Threads.nthreads())
p = 1000
n_batches = 100
a = CUDA.fill(1.0, (p, p, n_batches))
b = CUDA.fill(2.0, (p, p, n_batches))
c = CUDA.fill(0.0, (p, p, n_batches))


stream_test(a,b,c)
stream_test(a,b,c)

