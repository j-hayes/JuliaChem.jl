using Base.Threads
using LinearAlgebra
using CUDA
using TensorOperations
using JuliaChem.Shared.Constants.SCF_Keywords
using JuliaChem.Shared
using Serialization
using HDF5
using ThreadPinning
using Profile
using FileIO

@inline function twoD_to1Dindex(i, j, p)
    return (i - 1) * p + j
end


function get_screening_metadata!(scf_data, jeri_engine_thread, two_center_integrals, basis_sets)

    max_P_P = get_max_P_P(two_center_integrals)

    scf_data.screening_data.shell_screen_matrix,
    scf_data.screening_data.basis_function_screen_matrix,
    scf_data.screening_data.sparse_pq_index_map =
        schwarz_screen_itegrals_df(scf_data, 10^-12, max_P_P, basis_sets, jeri_engine_thread)
    basis_function_screen_matrix = scf_data.screening_data.basis_function_screen_matrix


    scf_data.screening_data.non_screened_p_indices_count = zeros(Int64, scf_data.μ)
    scf_data.screening_data.non_zero_coefficients = Vector{Array}(undef, scf_data.μ)
    scf_data.screening_data.screened_indices_count = sum(basis_function_screen_matrix)
    scf_data.screening_data.sparse_p_start_indices = zeros(Int64, scf_data.μ)

    non_screened_p_indices_count = scf_data.screening_data.non_screened_p_indices_count
    non_zero_coefficients = scf_data.screening_data.non_zero_coefficients
    sparse_pq_index_map = scf_data.screening_data.sparse_pq_index_map


    Threads.@threads for pp in 1:scf_data.μ
        first_index = 1
        while scf_data.screening_data.sparse_p_start_indices[pp] == 0
            if scf_data.screening_data.basis_function_screen_matrix[first_index, pp] != 0
                scf_data.screening_data.sparse_p_start_indices[pp] = sparse_pq_index_map[first_index, pp]
                break
            end
            first_index += 1
        end

        non_screened_p_indices_count[pp] = sum(view(basis_function_screen_matrix, :, pp))
        non_zero_coefficients[pp] = zeros(scf_data.occ, non_screened_p_indices_count[pp]) #todo justm aket his a vector and use it based on indicies calculated
    end
    # println("sparse_pq_index_map")
    # display(scf_data.screening_data.sparse_pq_index_map)
    # println("sparse_p_start_indices")
    # display(scf_data.screening_data.sparse_p_start_indices)
end


function get_triangle_matrix_length(n)::Int
    return n * (n + 1) ÷ 2
end

function df_rhf_fock_build_screened!(scf_data, jeri_engine_thread_df::Vector{T}, jeri_engine_thread::Vector{T2},
    basis_sets::CalculationBasisSets,
    occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine,T2<:RHFTEIEngine}
    comm = MPI.COMM_WORLD
    indicies = get_df_static_basis_indices(basis_sets, MPI.Comm_size(comm), MPI.Comm_rank(comm))

    p = scf_data.μ
    q = scf_data.μ
    Q = scf_data.A
    occ = scf_data.occ

    occupied_orbital_coefficients = permutedims(occupied_orbital_coefficients, (2, 1))

    if iteration == 1

        two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
        println("screening metadata")
        @time get_screening_metadata!(scf_data, jeri_engine_thread, two_center_integrals, basis_sets)


        load = scf_options.load
        scf_options.load = "screened" #todo remove
        println("three center integrals ")
        @time three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options, scf_data)
        println("permute three center integrals")
        @time three_center_integrals = permutedims(three_center_integrals, (2, 1)) #todo put this in correct order in calulation step
        scf_options.load = load #todo remove


        # J_AB_invt = convert(Array, inv(cholesky(Hermitian(two_center_integrals, :L)).U))
        # invert two center integrals using blas  
        # two_center_integrals = transpose(two_center_integrals)
        println("J_AB_invt")
        @time begin
            LAPACK.potrf!('L', two_center_integrals)
            LAPACK.trtri!('L', 'N', two_center_integrals)
            J_AB_invt = two_center_integrals
        end
        # if MPI.Comm_size(MPI.COMM_WORLD) > 1
        #   J_AB_INV = J_AB_INV[:,indicies]
        # end

        println("form B")
        # @time BLAS.trmm!('L', 'L', 'N', 'N', 1.0, J_AB_invt, scf_data.D)
        scf_data.D = zeros(size(three_center_integrals))
        @time BLAS.gemm!('N', 'N', 1.0, J_AB_invt, three_center_integrals, 0.0, scf_data.D)
        three_center_integrals = zeros(0)
        println("allocate memory")
        scf_data.D_tilde = zeros(Float64, (scf_data.A, scf_data.occ, scf_data.μ))
        scf_data.J = zeros(Float64, scf_data.screening_data.screened_indices_count)
        scf_data.K = zeros(Float64, size(scf_data.two_electron_fock))
        scf_data.coulomb_intermediate = zeros(Float64, scf_data.A)
        scf_data.density_array = zeros(Float64, scf_data.screening_data.screened_indices_count)
        scf_data.D_triangle = zeros(0)
        scf_data.two_electron_fock_triangle = zeros(0)
        scf_data.two_electron_fock_GPU = zeros(0)
        scf_data.thread_two_electron_fock = zeros(0)

        D_size = Base.summarysize(scf_data.D) / 1024^2
        essential_size = Base.summarysize(scf_data.D_tilde) / 1024^2
        essential_size += Base.summarysize(scf_data.coulomb_intermediate) / 1024^2
        essential_size += D_size

        # essential_size += Base.summarysize(scf_data.J) / 1024^2

        scfdata_size = Base.summarysize(scf_data) / 1024^2
        println("scf_data size = ", scfdata_size, " MB")
        println("essential_size size = ", essential_size, " MB")
        println("scf_data.D size = ", D_size, " MB")
    end
    println("exchange")
    @time calculate_exchange_screened!(scf_data, occupied_orbital_coefficients)
    println("coulomb")
    @time calculate_coulomb_screened(scf_data, occupied_orbital_coefficients)
    flush(stdout)

end

function calculate_exchange_screened!(scf_data, occupied_orbital_coefficients)
    p = scf_data.μ
    Q = scf_data.A
    occ = scf_data.occ
    M = scf_data.A
    N = scf_data.occ
    alpha = 1.0
    beta = 0.0

    linear_indicesB = LinearIndices(scf_data.D)
    linear_indicesW = LinearIndices(scf_data.D_tilde)

    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    # get number of threads 
    dynamic_p = Threads.nthreads() + 1

    dynamic_lock = Threads.ReentrantLock()

    @time begin
        Threads.@sync for thread in 1:Threads.nthreads()
            Threads.@spawn begin
                pp = thread
                non_zero_r_index = 1
                K = 1
                while pp <= p
                    non_zero_r_index = 1
                    for r in 1:p
                        if scf_data.screening_data.basis_function_screen_matrix[r, pp]
                            scf_data.screening_data.non_zero_coefficients[pp][:, non_zero_r_index] .= view(occupied_orbital_coefficients, :, r)
                            non_zero_r_index += 1
                        end
                    end
                    K = scf_data.screening_data.non_screened_p_indices_count[pp]
                    A_ptr = pointer(scf_data.D, linear_indicesB[1, scf_data.screening_data.sparse_p_start_indices[pp]])
                    B_ptr = pointer(scf_data.screening_data.non_zero_coefficients[pp], 1)
                    C_ptr = pointer(scf_data.D_tilde, linear_indicesW[1, 1, pp])
                    call_gemm!(Val(false), Val(true), M, N, K, alpha, A_ptr, B_ptr, beta, C_ptr)

                    lock(dynamic_lock) do
                        if dynamic_p <= scf_data.μ
                            pp = dynamic_p
                            dynamic_p += 1
                        else
                            pp = scf_data.μ + 1
                        end
                    end
                end


            end
        end
    end #time
    BLAS.set_num_threads(blas_threads)
    
    BLAS.gemm!('T', 'N', -1.0, reshape(scf_data.D_tilde, (Q * occ, p)), reshape(scf_data.D_tilde, (Q * occ, p)), 0.0, scf_data.two_electron_fock)
    # call_gemm!(Val(true), Val(false), p, p, occ*Q, -1.0, pointer(scf_data.D_tilde, 1), pointer(scf_data.D_tilde, 1), 0.0, pointer(scf_data.two_electron_fock,1))
    # if scf_data.μ < 400
    #     BLAS.gemm!('T', 'N', -1.0, reshape(scf_data.D_tilde, (Q * occ, p)), reshape(scf_data.D_tilde, (Q * occ, p)), 0.0, scf_data.two_electron_fock)
    # else
    #     @time calculate_K_upper_diagonal_block(scf_data)
    # end
end

function calculate_coulomb_screened(scf_data, occupied_orbital_coefficients)
    BLAS.gemm!('T', 'N', 1.0, occupied_orbital_coefficients, occupied_orbital_coefficients, 0.0, scf_data.density)
    sparse_pq_index_map = scf_data.screening_data.sparse_pq_index_map
    basis_function_screen_matrix = scf_data.screening_data.basis_function_screen_matrix

    Threads.@threads for index in CartesianIndices(scf_data.density)
        if !basis_function_screen_matrix[index[1], index[2]] 
            continue
        end            
        # scf_data.density_array[sparse_pq_index_map[index[1], index[2]]] = scf_data.density[index]
        # doing symm
        

        if index[1] != index[2]
            scf_data.density_array[sparse_pq_index_map[index[1], index[2]]] = 2.0*scf_data.density[index]
        else
            scf_data.density_array[sparse_pq_index_map[index[1], index[2]]] = scf_data.density[index]
        end
    end
    sparse_p_start_indices = scf_data.screening_data.sparse_p_start_indices
    # non_screened_p_indices_count = scf_data.screening_data.non_screened_p_indices_count
    p = scf_data.μ
    scf_data.coulomb_intermediate .= 0.0
    # println(scf_data.screening_data.sparse_p_start_indices)
    println("length density array" , length(scf_data.density_array))
    for pp in 1:(p-1)
        range_start = sparse_pq_index_map[pp, pp]
        range_end = scf_data.screening_data.sparse_p_start_indices[pp+1] 
        range_end -= 1
        
        # range = range_start:range_end
        BLAS.gemv!('N', 1.0, 
        view(scf_data.D, :, range_start:range_end), 
        view(scf_data.density_array,  range_start:range_end),
        1.0, scf_data.coulomb_intermediate)
    end
    BLAS.gemv!('N', 1.0, 
     view(scf_data.D, :, size(scf_data.D, 2)),
     view(scf_data.density_array, length(scf_data.density_array):length(scf_data.density_array)),
      1.0, scf_data.coulomb_intermediate)

    # println(scf_data.coulomb_intermediate)
    # println("done with coulomb intermediate")
    # flush(stdout)
    # BLAS.gemv!('N', 1.0, scf_data.D, scf_data.density_array, 1.0, scf_data.coulomb_intermediate)

    # do symm J 
    scf_data.J .= 0.0
    for pp in 1:(p-1)
        range_start = sparse_pq_index_map[pp, pp]
        range_end = scf_data.screening_data.sparse_p_start_indices[pp+1]
        range_end -= 1
        BLAS.gemv!('T', 2.0,
            view(scf_data.D, :, range_start:range_end),
            scf_data.coulomb_intermediate,
            1.0, view(scf_data.J, range_start:range_end))
    end
    BLAS.gemv!('T', 2.0,
        view(scf_data.D, :, size(scf_data.D, 2)),
        scf_data.coulomb_intermediate,
        1.0, view(scf_data.J, length(scf_data.J):length(scf_data.J)))

    
    

    # display(scf_data.J[1:1000])

    # BLAS.gemv!('T', 2.0, scf_data.D, scf_data.coulomb_intermediate, 0.0, scf_data.J)

    Threads.@threads for index in CartesianIndices(scf_data.two_electron_fock)
        if !basis_function_screen_matrix[index] || index[2] > index[1]
            continue
        end
        scf_data.two_electron_fock[index] += scf_data.J[sparse_pq_index_map[index]]
        if index[1] != index[2]
            scf_data.two_electron_fock[index[2], index[1]] = scf_data.two_electron_fock[index]
        end
    end
end


function twoD_toLower_triangular(i, j, n)
    if j <= i
        return i + (2 * n - j) * (j - 1) ÷ 2
    else
        return j + (2 * n - i) * (i - 1) ÷ 2
    end
end

function calculate_K_upper_diagonal_block(scf_data)
    W = scf_data.D_tilde
    p = scf_data.μ
    Q = scf_data.A
    occ = scf_data.occ
    # basis_function_screen_matrix = scf_data.screening_data.basis_function_screen_matrix

    batch_width = 10
    # for i in 20:-1:10
    #     if p%i == 0
    #         batch_width = p÷i
    #         break
    #     end
    # end
    batch_size = p ÷ batch_width
    p = size(W, 3)
    occ = size(W, 2)
    Q = size(W, 1)
  
    transA = true
    transB = false
    alpha = -1.0
    beta = 0.0
    linear_indices = LinearIndices(W)

    M = batch_size
    N = batch_size
    K = Q * occ

    batch_indexes = zeros(Int, batch_width, batch_width)
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    if length(scf_data.k_blocks) == 0
        scf_data.k_blocks = zeros(Float64, batch_size, batch_size, batch_width*batch_width)
        #todo just make enough blocks for the number of threads
    end
    exchange_blocks = scf_data.k_blocks
    K_linear_indices = LinearIndices(exchange_blocks)

    symm = true

 
    number_of_batches = batch_width^2
    batch_cartesian_indices = CartesianIndices((batch_width, batch_width))
    if symm
        # batch_cartesian_indices = [(i, j) for i in 1:batch_width, j in 1:i]
        # number_of_batches = get_triangle_matrix_length(batch_width)
    end
    n_threads = Threads.nthreads()
    dynamic_batch_index = n_threads + 1
    dyanmic_lock = Threads.ReentrantLock()
    println("K block")
        @sync for thread in 1:n_threads
            Threads.@spawn begin
                batch_index = thread
                while batch_index <= number_of_batches

                    # batch_index = twoD_to1Dindex(i, j, batch_width)
                    i = batch_cartesian_indices[batch_index][1]
                    j = batch_cartesian_indices[batch_index][2]

                    if symm && j > i #add back for symm
                        lock(dyanmic_lock) do
                            if dynamic_batch_index <= batch_width^2
                                batch_index = dynamic_batch_index
                                dynamic_batch_index += 1
                            else
                                batch_index = number_of_batches + 1
                            end
                        end
                        continue
                    end

                    p_range = (i-1)*batch_size+1:i*batch_size
                    p_start = (i - 1) * batch_size + 1

                    q_range = (j-1)*batch_size+1:j*batch_size
                    q_start = (j - 1) * batch_size + 1

                    # total_non_screened_indices = sum(
                    #     view(scf_data.screening_data.basis_function_screen_matrix, p_range, q_range))

                    # if total_non_screened_indices == 0
                    #     # println("block is screend i $i j $j")
                    #     continue
                    # end



                    A_ptr = pointer(W, linear_indices[1, 1, p_start])
                    B_ptr = pointer(W, linear_indices[1, 1, q_start])
                    C_ptr = pointer(exchange_blocks, K_linear_indices[1, 1, batch_index])

                    call_gemm!(Val(transA), Val(transB), M, N, K, alpha, A_ptr, B_ptr, beta, C_ptr)

                    scf_data.two_electron_fock[p_range, q_range] .= view(exchange_blocks, :, :, batch_index)
                    if symm && i != j
                        scf_data.two_electron_fock[q_range, p_range] .= transpose(view(exchange_blocks, :, :, batch_index))
                    end
                    lock(dyanmic_lock) do
                        if dynamic_batch_index <= batch_width^2
                            batch_index = dynamic_batch_index
                            dynamic_batch_index += 1
                        else
                            batch_index = number_of_batches + 1
                        end
                    end
                end
            end
        end#sync
    BLAS.set_num_threads(blas_threads)


    #non square part of K not include in the blocks above if any are non screened put back after full block is working
    # q_range = p-(p%block_size-1):p
    # p_range = 1:p
    # if length(q_range) > 0
    #   number_of_non_screened_pq_pairs = sum(basis_function_screen_matrix[q_range,p_range])
    #   if number_of_non_screened_pq_pairs != 0
    #     BLAS.gemm!('T', 'N', -1.0, view(W, :, p_range), view(W, :, q_range),
    #       0.0, view(K, p_range, q_range))
    #   end
    # end

    # Threads.@threads for pp in 1:p
    #   for qq in 1:pp-1
    #     K[pp, qq] = K[qq, pp]
    #   end
    # end
    # axpy!(1.0, scf_data.K, scf_data.two_electron_fock)
end


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

function calculate_W_screened(B, p, occ, occupied_orbital_coefficients, W, r_ranges, B_ranges)
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    transA = false
    transB = true
    alpha = 1.0
    beta = 1.0
    Q = size(B, 1)


    B_linear_indices = LinearIndices(B)
    W_linear_indices = LinearIndices(W)
    ooc_linear_indicies = LinearIndices(occupied_orbital_coefficients)
    M = Q
    N = occ



    r_index_count = 0
    all_ranges = Array{Tuple{Int,Int},1}()
    for pp in 1:p
        for r_index in 1:length(r_ranges[pp])
            r_index_count += 1
            push!(all_ranges,)
        end
    end

    # order p by length of r_ranges

    p_order = sort(1:p, by=x -> length(r_ranges[x]), rev=true)
    # range_lengths = sort([length(r_ranges[pp]) for pp in 1:p], rev = true)
    # p_timings = zeros(Float64, p,2)

    println("do W loop")
    dynamic_p_index = n_threads + 1
    dynamic_p_lock = Threads.ReentrantLock()

    elapsed_times = zeros(Float64, p)

    Threads.@sync for threadindex in 1:n_threads
        Threads.@spawn begin
            # for p_sorted_indec in threadindex:n_threads:p # loop over p balanced to each threadindex 
            #   pp =  p_order[p_sorted_indec]
            pp = p_order[threadindex]
            while pp <= p
                fill!(view(W, :, :, pp), 0.0)
                K = 0
                A_ptr = pointer(B, B_linear_indices[1, B_ranges[pp][1][1]])
                B_ptr = pointer(occupied_orbital_coefficients, ooc_linear_indicies[1, r_ranges[pp][1][1]])
                C_ptr = pointer(W, W_linear_indices[1, 1, pp])
                for index in eachindex(r_ranges[pp])
                    K = r_ranges[pp][index][end] - r_ranges[pp][index][1] + 1
                    A_ptr = pointer(B, B_linear_indices[1, B_ranges[pp][index][1]])
                    B_ptr = pointer(occupied_orbital_coefficients, ooc_linear_indicies[1, r_ranges[pp][index][1]])
                    call_gemm!(Val(transA), Val(transB), M, N, K, alpha, A_ptr, B_ptr, beta, C_ptr)
                end #inner loop r_ranges
                lock(dynamic_p_lock) do
                    if dynamic_p_index <= p
                        pp = p_order[dynamic_p_index]
                        dynamic_p_index += 1
                    else
                        pp = p + 1 # break out of while loop
                    end
                end
            end #while pp <= p/for pp in threadindex:n_threads:p
        end#spawn
    end#sync
    #put all values in an array where elapses where not == 0
    println("done W loop")
    # save("/home/ac.jhayes/source/JuliaChem.jl/W_C40H82_56_psorted_threads_jlseskylake.jlprof", Profile.retrieve()...)
    BLAS.set_num_threads(blas_threads)
end
