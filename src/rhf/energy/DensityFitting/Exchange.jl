function calculate_dfrhf_exchange!(scf_data::SCFData, scf_options::SCFOptions, occupied_orbital_coefficients::Array{T,2}, jc_timing::JCTiming) where {T<:Union{Float32, Float64}}
    use_screening = do_dfrhf_screening(scf_options)
    if use_screening
        W_time = @elapsed calculate_dfrhf_W_screened!(scf_data, occupied_orbital_coefficients)
    else
        W_time = @elapsed calculate_dfrhf_W_noscreen!(scf_data, occupied_orbital_coefficients)
    end

    if scf_options.df_use_K_sym
       K_time = @elapsed calculate_dfrhf_exchange_sym!(scf_data, scf_options, jc_timing)
    else
       K_time = @elapsed calculate_dfrhf_exchange_no_sym!(scf_data, occupied_orbital_coefficients)
    end
end


function calculate_dfrhf_exchange_sym!(scf_data::SCFData, scf_options::SCFOptions, jc_timing::JCTiming)
    p = scf_data.μ
    occ = scf_data.occ

    K_block_width = scf_data.screening_data.K_block_width
    
    transA = true
    transB = false
    alpha = -1.0
    # first Q index beta = 0, subsequent has beta = 1 
    beta = 0.0

    M = K_block_width
    N = K_block_width
    
   
    n_threads = Threads.nthreads()
    
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    
    exchange_blocks = scf_data.k_blocks
    K_linear_indices = LinearIndices(exchange_blocks)
    K_non_square_block_linear_indices = LinearIndices(scf_data.k_non_square_blocks)
    lower_triangle_length = get_triangle_matrix_length(scf_options.df_exchange_n_blocks)
    use_non_square_blocks = p % scf_options.df_exchange_n_blocks != 0

    n_non_square_blocks = min(lower_triangle_length % n_threads, p % K_block_width)
    index_times = zeros(Float64, lower_triangle_length + n_non_square_blocks)


    k_block_p_limit = K_block_width * scf_options.df_exchange_n_blocks
    k_block_q_limit = k_block_p_limit

    scf_data.two_electron_fock .= 0.0 #zero out Fock matrix
    num_Q_ranges = length(scf_data.W_batches)
    #loop over the batches of Q ranges
    for Q_range_index in 1:num_Q_ranges
        W = scf_data.W_batches[Q_range_index]
        Q = size(W, 1) 
        K = Q * occ
        linear_indices = LinearIndices(W)
        Threads.@threads for index in lower_triangle_length:-1:1 
            pp, qq = scf_data.screening_data.exchange_batch_indexes[index]

            p_start = (pp - 1) * K_block_width + 1
            p_range = p_start:pp*K_block_width
            if p_range[end] == k_block_p_limit
                p_range = p_start:p
            end
            
            q_start = (qq - 1) * K_block_width + 1
            q_range = q_start:qq*K_block_width
            if q_range[end] == k_block_q_limit
                q_range = q_start:p
            end

            # Will be changing what W_index is
            # pointers to place in memory where matrix multiplication will be done
            A_ptr = pointer(W, linear_indices[1, 1, p_start])
            B_ptr = pointer(W, linear_indices[1, 1, q_start])

            if p_range[end] == p && q_range[end] == p && use_non_square_blocks #bottom corner block
                C_ptr = pointer(scf_data.bottom_corner_k_block, 1)
                call_gemm!(Val(transA), Val(transB), length(p_range), length(q_range), K, alpha, A_ptr, B_ptr, beta, C_ptr)
                # will be .+=, need to zero out Fock matrix in first iteration at beginning of function
                scf_data.two_electron_fock[p_range, q_range] .+= scf_data.bottom_corner_k_block
            elseif (p_range[end] == p || q_range[end] == p) && use_non_square_blocks #non square block
                C_ptr = pointer(scf_data.k_non_square_blocks, K_non_square_block_linear_indices[1,1, qq])
                C_block = view(scf_data.k_non_square_blocks, :, :, qq)
                call_gemm!(Val(transA), Val(transB), length(p_range), length(q_range), K, alpha, A_ptr, B_ptr, beta, C_ptr)
                scf_data.two_electron_fock[p_range, q_range] .+= C_block
                if pp != qq
                    scf_data.two_electron_fock[q_range, p_range] .+= transpose(C_block) 
                end
            else #square block (normal)
                C_ptr = pointer(exchange_blocks, K_linear_indices[1, 1, index])
                call_gemm!(Val(transA), Val(transB), M, N, K, alpha, A_ptr, B_ptr, beta, C_ptr)
                scf_data.two_electron_fock[p_range, q_range] .+= view(exchange_blocks, :,:, index)
                if pp != qq
                    scf_data.two_electron_fock[q_range, p_range] .+= transpose(view(exchange_blocks, :,:, index)) 
                end
            end
        end#sync
    end

    BLAS.set_num_threads(blas_threads)
    
end


function calculate_dfrhf_W_noscreen!(scf_data::SCFData, occupied_orbital_coefficients::Array{T,2}) where {T<:Union{Float32, Float64}}
    B = scf_data.B
    W = scf_data.W_batches
    p = scf_data.μ # number of basis functions
    n_ooc = scf_data.occ # number of occupied orbitals
    one_mixed = T(1.0)
    zero_mixed = T(0.0)

    num_Q_ranges = size(B, 1)
    for Q_range_index in 1:num_Q_ranges
        Q_size = size(scf_data.W_batches[Q_range_index],1)
        B_reshape = reshape(B[Q_range_index], (Q_size*p, p))
        W_reshape = reshape(W[Q_range_index], (n_ooc,Q_size*p))

        BLAS.gemm!('N', 'T', one_mixed, occupied_orbital_coefficients, B_reshape, zero_mixed, W_reshape)
    end
end

function calculate_dfrhf_exchange_no_sym!(scf_data, occupied_orbital_coefficients::Array{T,2}) where {T<:Union{Float32, Float64}}

    M = scf_data.μ 
    N = scf_data.μ
    num_Q_ranges = size(scf_data.B,1)
    float_type = typeof(scf_data.B[1][1,1])
    alpha = float_type(-1.0)
    beta = float_type(0.0)
    left_transpose = true
    right_transpose = false

    for Q_range_index in 1:num_Q_ranges
        # zero out two_electron_fock on first iteration only
        # if Q_range_index > 1
        #     beta = float_type(1.0)
        # end do this if we put back C_ptr = pointer(scf_data.two_electron_fock, 1) in call_gemm!

        K = size(scf_data.W_batches[Q_range_index],1)*scf_data.occ

        A_ptr = pointer(scf_data.W_batches[Q_range_index], 1)
        B_ptr = pointer(scf_data.W_batches[Q_range_index], 1)
        C_ptr = pointer(scf_data.K[Q_range_index], 1)

        call_gemm!(Val(left_transpose), Val(right_transpose), M, N, K, alpha, A_ptr, B_ptr, beta, C_ptr) 
    end
    #remove this if we put back C_ptr = pointer(scf_data.two_electron_fock, 1) in call_gemm!
    scf_data.two_electron_fock .= 0.0
    for Q_range_index in 1:num_Q_ranges
        # print the first 10 values of the K matrix for each batch
        scf_data.two_electron_fock += scf_data.K[Q_range_index]
    end

    # println("exchange 2: ")
    # display(scf_data.two_electron_fock)

end


function calculate_dfrhf_W_screened!(scf_data::SCFData, occupied_orbital_coefficients::Array{T,2}) where {T<:Union{Float32, Float64}}

    p = scf_data.μ # number of basis functions
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1) # use one thread for BLAS
    n_threads = Threads.nthreads() #getting number of available threads
    W = scf_data.W_batches
    num_Q_ranges = size(scf_data.B,1)
    N = scf_data.occ
    alpha = T(1.0)
    beta = T(0.0)
    

    # builds non-screened coefficient matrix for each primary basis index (see Huang et al.) "To compute W in (4) ..."
    # Threads.@threads for pp in 1:p
    for pp in 1:p
        non_zero_r_index = 1
        for r in 1:p
            if scf_data.screening_data.basis_function_screen_matrix[r, pp]
                scf_data.non_zero_coefficients[pp][:, non_zero_r_index] .= view(occupied_orbital_coefficients, :, r)
                non_zero_r_index += 1
            end
        end
    end

    # for every Q range, calculate Q_ranges contribution to W matrix
    for Q_range_index in 1:num_Q_ranges
        M = size(scf_data.B[Q_range_index],1)
        linear_indicesB = LinearIndices(scf_data.B[Q_range_index])
        linear_indicesW = LinearIndices(W[Q_range_index])
        Threads.@threads for pp in 1:p
            K = scf_data.screening_data.non_screened_p_indices_count[pp]
            A_ptr = pointer(scf_data.B[Q_range_index], linear_indicesB[1, scf_data.screening_data.sparse_p_start_indices[pp]])
            B_ptr = pointer(scf_data.non_zero_coefficients[pp], 1)
            C_ptr = pointer(W[Q_range_index], linear_indicesW[1, 1, pp])
            call_gemm!(Val(false), Val(true), M, N, K, alpha, A_ptr, B_ptr, beta, C_ptr)
        end
    end
    BLAS.set_num_threads(blas_threads)
    
end


#todo move this to a BLAS functions file
function call_gemm!(transA::Val, transB::Val,
    M::Int, N::Int, K::Int,
    alpha::T, A::Ptr{T}, B::Ptr{T},
    beta::T, C::Ptr{T}) where {T<:Union{Float32, Float64}}

    #TODO pass bool instead of val and wrap val in this function 
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

    #if T is Float32, use the single precision BLAS function
    if T == Float32
        ccall((sgemm_32_, BLAS.libblas), Nothing,
        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
            Ref{BlasInt}, Ref{T}, Ptr{T}, Ref{BlasInt},
            Ptr{T}, Ref{BlasInt}, Ref{T}, Ptr{T},
            Ref{BlasInt}),
        convtrans(transA), convtrans(transB), M, N, K,
        alpha, A, lda, B, ldb, beta, C, ldc)   
    elseif T == Float64
        ccall((dgemm_64_, BLAS.libblas), Nothing,
        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
            Ref{BlasInt}, Ref{T}, Ptr{T}, Ref{BlasInt},
            Ptr{T}, Ref{BlasInt}, Ref{T}, Ptr{T},
            Ref{BlasInt}),
        convtrans(transA), convtrans(transB), M, N, K,
        alpha, A, lda, B, ldb, beta, C, ldc)
    end
    # conditionally/
    
end

function setup_dfrhf_exchange_blocks!(scf_data::SCFData, scf_options::SCFOptions, jc_timing::JCTiming)

    K_block_width = 0
    lower_triangle_length = get_triangle_matrix_length(scf_options.df_exchange_n_blocks)

    if scf_data.μ < 100 #if the # of basis functions is small just do a dense calculation with one block
        K_block_width = scf_data.μ
        scf_options.df_exchange_n_blocks = 1
    else
        K_block_width = scf_data.μ ÷ scf_options.df_exchange_n_blocks
    end
    scf_data.screening_data.K_block_width = K_block_width


    scf_data.k_blocks = zeros(Float64, K_block_width, K_block_width, lower_triangle_length)

    the_batch_index = 1
    exchange_batch_indexes = Array{Tuple{Int, Int}}(undef, lower_triangle_length)
    for iii in 1:scf_options.df_exchange_n_blocks
        for jjj in 1:iii
            exchange_batch_indexes[the_batch_index] = (iii, jjj)
            the_batch_index+=1
        end
    end
      # println("scf_data.μ % scf_options.df_exchange_n_blocks = ", scf_data.μ % scf_options.df_exchange_n_blocks)
    if scf_data.μ % scf_options.df_exchange_n_blocks != 0
        non_square_size = K_block_width + scf_data.μ % scf_options.df_exchange_n_blocks
        scf_data.k_non_square_blocks = zeros(Float64, non_square_size, K_block_width, scf_options.df_exchange_n_blocks)
        scf_data.bottom_corner_k_block = zeros(Float64, non_square_size, non_square_size)
    end

    scf_data.screening_data.exchange_batch_indexes = exchange_batch_indexes

    jc_timing.non_timing_data[JCTC.total_exchange_blocks] = string(lower_triangle_length)
    jc_timing.non_timing_data[JCTC.df_exchange_n_blocks] = string(scf_options.df_exchange_n_blocks)
end

# Currently not used, to be brought back if this feature is resurected later if a 
# appropriate method for systematically manipulating the input geometry such that it maximizes the 
# number of screened blocks is found.
function screen_exchange_blocks(lower_triangle_length::Int, scf_data::SCFData, scf_options::SCFOptions, jc_timing::JCTiming)
    while block_index <= lower_triangle_length
        pp, qq = exchange_batch_indexes[block_index]
        if scf_options.df_screen_exchange     
            p_range = (pp-1)*K_block_width+1:pp*K_block_width
            q_range = (qq-1)*K_block_width+1:qq*K_block_width
            total_non_screened_indices = sum(
                view(scf_data.screening_data.basis_function_screen_matrix, p_range, q_range))
            if total_non_screened_indices != 0 #skip where all are screened
                push!(blocks_to_calculate, block_index) 
                block_screen_matrix[pp, qq] = true
            end
        else
            push!(blocks_to_calculate, block_index) 
            block_screen_matrix[pp, qq] = true
        end
        block_index += 1
    end
    scf_data.screening_data.block_screen_matrix = block_screen_matrix
    scf_data.screening_data.blocks_to_calculate = blocks_to_calculate

    jc_timing.non_timing_data[JCTC.unscreened_exchange_blocks] = string(length(blocks_to_calculate))

end