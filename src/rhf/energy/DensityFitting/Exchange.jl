function calculate_dfrhf_exchange!(scf_data::SCFData, scf_options::SCFOptions, occupied_orbital_coefficients::Array{T,2}, jc_timing::JCTiming) where {T<:Union{Float32, Float64}}
    W_time = @elapsed calculate_W_screened!(scf_data, occupied_orbital_coefficients)
    scf_options.df_use_K_sym = false
    if scf_options.df_use_K_sym
       error("Not imlemented yet")
    else
        K_time = @elapsed calculate_dfrhf_exchange_no_sym!(scf_data, occupied_orbital_coefficients)
    end
end


function calculate_dfrhf_exchange_no_sym!(scf_data, occupied_orbital_coefficients::Array{T,2}) where {T<:Union{Float32, Float64}}

    M = scf_data.μ 
    N = scf_data.μ
    num_Q_ranges = size(scf_data.B,1)
    float_type = typeof(scf_data.B[1][1,1])
    alpha = float_type(-1.0)
    beta = float_type(0.0)

    for Q_range_index in 1:num_Q_ranges
        # zero out two_electron_fock on first iteration only
        # if Q_range_index > 1
        #     beta = float_type(1.0)
        # end do this if we put back C_ptr = pointer(scf_data.two_electron_fock, 1) in call_gemm!

        K = size(scf_data.W_batches[Q_range_index],1)*scf_data.occ

        A_ptr = pointer(scf_data.W_batches[Q_range_index], 1)
        B_ptr = pointer(scf_data.W_batches[Q_range_index], 1)
        C_ptr = pointer(scf_data.K[Q_range_index], 1)

        call_gemm!(Val(true), Val(false), M, N, K, alpha, A_ptr, B_ptr, beta, C_ptr) 
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


function calculate_W_screened!(scf_data::SCFData, occupied_orbital_coefficients::Array{T,2}) where {T<:Union{Float32, Float64}}

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



