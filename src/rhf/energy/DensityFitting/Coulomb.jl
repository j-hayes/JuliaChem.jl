function calculate_dfrhf_coulomb!(scf_data::SCFData, scf_options::SCFOptions, 
        occupied_orbital_coefficients::Array{T,2}, jc_timing::JCTiming, iteration::Int) where {T<:Union{Float32, Float64}}
    calculate_density!(scf_data, scf_options, occupied_orbital_coefficients, jc_timing, iteration)
    if scf_options.df_use_J_sym
        calculate_dfrhf_coulomb_sym!(scf_data, scf_options, jc_timing, iteration)
    else
        calculate_dfrhf_coulomb_no_sym!(scf_data, scf_options, jc_timing, iteration)
    end
end


function calculate_dfrhf_coulomb_sym!(scf_data::SCFData, scf_options::SCFOptions, jc_timing::JCTiming, iteration::Int)
    calculate_dfrhf_coulomb_intermediate_sym!(scf_data, scf_options, jc_timing, iteration)
    calculate_dfrhf_J_sym!(scf_data, scf_options, jc_timing, iteration)
end

function calculate_dfrhf_coulomb_no_sym!(scf_data::SCFData, scf_options::SCFOptions , jc_timing::JCTiming, iteration::Int) 
    V_time = 0.0
    J_time = 0.0
    J_copy_time = 0.0
    one = scf_options.contraction_float_type(1.0)
    two = scf_options.contraction_float_type(2.0)
    zero = scf_options.contraction_float_type(0.0)
    for Q_range_index in 1:size(scf_data.B, 1)
      V_time += @elapsed BLAS.gemv!('N', one, scf_data.B[Q_range_index], scf_data.density_array, zero, scf_data.V_batches[Q_range_index])
      J_time += @elapsed BLAS.gemv!('T', two, scf_data.B[Q_range_index], scf_data.V_batches[Q_range_index], zero, scf_data.J[Q_range_index])
      J_copy_time += @elapsed copy_screened_coulomb_to_fock!(scf_data, scf_data.J[Q_range_index], scf_data.two_electron_fock)
    end

    jc_timing.timings[JCTiming_key(JCTC.V_time, iteration)] = V_time
    jc_timing.timings[JCTiming_key(JCTC.J_time, iteration)] = J_time
    jc_timing.timings[JCTiming_key(JCTC.copy_J_time, iteration)] = J_copy_time
end


function calculate_coulomb_symmetric_range(scf_data::SCFData,screening_data::ScreeningData, scf_options::SCFOptions, pp::Int64)
    if do_dfrhf_screening(scf_options)
        if pp == scf_data.μ
            return scf_data.screening_data.sparse_p_start_indices[pp]:scf_data.screening_data.screened_indices_count
        end
        range_start = scf_data.screening_data.sparse_p_start_indices[pp]
        range_end = scf_data.screening_data.sparse_p_start_indices[pp+1] - 1
        return range_start:range_end
    else
        return (pp-1)*scf_data.μ + 1:pp*scf_data.μ
    end
end

function calculate_dfrhf_coulomb_intermediate_sym!(scf_data::SCFData, scf_options::SCFOptions, jc_timing::JCTiming, iteration::Int)
    sparse_pq_index_map = scf_data.screening_data.sparse_pq_index_map
    blas_threads = BLAS.get_num_threads()
    alpha = scf_options.contraction_float_type(1.0)
    beta = scf_options.contraction_float_type(0.0)
    V_time = @elapsed begin 
        last_blas_add_time = 0.0
        p = scf_data.μ
        BLAS.set_num_threads(1)
        n_threads = min(Threads.nthreads(), p-1)
        num_p_per_thread = p ÷ n_threads

        num_Q_ranges = length(scf_data.W_batches)
        for Q_range_index in 1:num_Q_ranges
            B = scf_data.B[Q_range_index]
            W = scf_data.W_batches[Q_range_index]
            V = scf_data.V_batches[Q_range_index]
            rank_Q = size(B, 1)
            for tt in 1:n_threads
                p_thread_start = (tt - 1) * num_p_per_thread + 1
                p_thread_end =  tt * num_p_per_thread
                if tt == n_threads
                    p_thread_end = p-1
                end
                thread_V = view(view(W, :,:, p_thread_start), 1:rank_Q)
                beta = scf_options.contraction_float_type(0.0)
                for pp in p_thread_start:p_thread_end
                    if pp != p_thread_start
                        beta = scf_options.contraction_float_type(1.0)
                    end
                    range = calculate_coulomb_symmetric_range(scf_data,scf_data.screening_data, scf_options, pp)
                    BLAS.gemv!('N', alpha, 
                        view(B, :, range), 
                        view(scf_data.density_array, range),
                        beta, thread_V) 
                end
                if tt == n_threads  
                    last_range = calculate_coulomb_symmetric_range(scf_data,scf_data.screening_data, scf_options, scf_data.μ)
                    BLAS.gemv!('N', alpha, 
                    view(B, :, last_range),
                    view(scf_data.density_array, last_range),
                    scf_options.contraction_float_type(0.0), V)
                end
            end
            for t in 1:n_threads
                p_thread_start = (t - 1) * num_p_per_thread + 1
                axpy!(1.0, view(view(W, :,:, p_thread_start), 1:rank_Q), V)
            end 
        end
    end
    BLAS.set_num_threads(blas_threads)
    jc_timing.timings[JCTiming_key(JCTC.V_time, iteration)] = V_time
end

function calculate_dfrhf_J_sym!(scf_data::SCFData, scf_options::SCFOptions, jc_timing::JCTiming, iteration::Int)
    blas_threads = BLAS.get_num_threads()
    p = scf_data.μ
    alpha = scf_options.contraction_float_type(2.0)
    beta = scf_options.contraction_float_type(0.0)
    copy_J_time = 0.0
    J_time = @elapsed begin
        # do symm J 
        num_Q_ranges = length(scf_data.B)
        for Q_range_index in 1:num_Q_ranges
            V = scf_data.V_batches[Q_range_index]
            B = scf_data.B[Q_range_index]
            Threads.@threads for pp in 1:(p-1) #todo use call_gemv to remove view usage?
                range = calculate_coulomb_symmetric_range(scf_data,scf_data.screening_data, scf_options, pp)
                BLAS.gemv!('T', alpha,
                    view(B, :, range),
                    V,
                    beta, view(scf_data.J[Q_range_index], range))
                if pp == p-1
                    last_range = calculate_coulomb_symmetric_range(scf_data,scf_data.screening_data, scf_options, scf_data.μ)
                    BLAS.gemv!('T', alpha,
                        view(B, :, last_range),
                        V,
                        beta, view(scf_data.J[Q_range_index], last_range))
                end
            end
            copy_J_time += @elapsed copy_screened_coulomb_to_fock!(scf_data, scf_data.J[Q_range_index], scf_data.two_electron_fock)
        end
    end
    jc_timing.timings[JCTiming_key(JCTC.copy_J_time, iteration)] = copy_J_time
    jc_timing.timings[JCTiming_key(JCTC.J_time, iteration)] = J_time

    BLAS.set_num_threads(blas_threads)
end


function calculate_density!(scf_data::SCFData, scf_options::SCFOptions, 
    occupied_orbital_coefficients::Array{T,2}, jc_timing::JCTiming, iteration::Int) where {T<:Union{Float32, Float64}}
    blas_threads = BLAS.get_num_threads()

    if scf_data.μ < 1000
        BLAS.set_num_threads(1)
    end
    density_time = @elapsed begin
        BLAS.gemm!('T', 'N', T(1.0), occupied_orbital_coefficients, occupied_orbital_coefficients, T(0.0), scf_data.density)
        if do_dfrhf_screening(scf_options)
            copy_screened_density_to_array(scf_data)
        else
            scf_data.density_array = reshape(scf_data.density, scf_data.μ * scf_data.μ) # reshape the density matrix to a vector for gemv! density_array is used in the coulomb calculation
        end
    end
    jc_timing.timings[JCTiming_key(JCTC.density_time, iteration)] = density_time
    BLAS.set_num_threads(blas_threads)
end
