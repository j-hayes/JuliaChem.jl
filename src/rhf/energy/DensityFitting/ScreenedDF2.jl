

function do_dfrhf_screening(scf_options::SCFOptions)
    if scf_options.contraction_mode == Constants.SCF_Keywords.ContractionMode.screened || #screened CPU 
        scf_options.contraction_mode == Constants.SCF_Keywords.GPUAlgorithms.default # screened GPU
        return true
    end
    return false
end


function setup_dfrhf_screening!(scf_data::SCFData, scf_options::SCFOptions, jeri_engine_thread, two_center_integrals::Matrix{T}, 
    basis_sets::CalculationBasisSets, jc_timing::JCTiming) where {T<:Union{Float32, Float64}}

    if !do_dfrhf_screening(scf_options)
        return
    end
    
    sigma = scf_options.df_screening_sigma
    screening_time = @elapsed begin
        max_P_P = get_max_P_P(two_center_integrals)
        scf_data.screening_data.shell_screen_matrix,
        scf_data.screening_data.basis_function_screen_matrix,
        scf_data.screening_data.sparse_pq_index_map = schwarz_screen_itegrals_df(scf_data, sigma, max_P_P, basis_sets, jeri_engine_thread)
    end 

    screning_metadata_time = @elapsed begin 

        basis_function_screen_matrix = scf_data.screening_data.basis_function_screen_matrix
        scf_data.screening_data.non_screened_p_indices_count = zeros(Int64, scf_data.μ)
        scf_data.non_zero_coefficients = Vector{Array}(undef, scf_data.μ)
        scf_data.screening_data.screened_indices_count = sum(basis_function_screen_matrix)
        scf_data.screening_data.sparse_p_start_indices = zeros(Int64, scf_data.μ)
        scf_data.screening_data.non_zero_ranges = Vector{Array{UnitRange{Int}}}(undef, scf_data.μ)
        scf_data.screening_data.non_zero_sparse_ranges = Vector{Array{UnitRange{Int}}}(undef, scf_data.μ)

        Threads.@threads for pp in 1:scf_data.μ 
            first_index = 1
            while scf_data.screening_data.sparse_p_start_indices[pp] == 0
                if scf_data.screening_data.basis_function_screen_matrix[first_index, pp] != 0
                    scf_data.screening_data.sparse_p_start_indices[pp] = scf_data.screening_data.sparse_pq_index_map[first_index, pp]
                    break
                end
                first_index += 1
            end

            scf_data.screening_data.non_screened_p_indices_count[pp] = sum(view(basis_function_screen_matrix, :, pp))
            scf_data.non_zero_coefficients[pp] = zeros(T, (scf_data.occ, scf_data.screening_data.non_screened_p_indices_count[pp]))

            scf_data.screening_data.non_zero_ranges[pp] = Array{UnitRange{Int}}(undef, 0)
            scf_data.screening_data.non_zero_sparse_ranges[pp] = Array{UnitRange{Int}}(undef, 0)

            start_index = 0
            end_index = 0
            non_zero_index = 1
            for r in 1:scf_data.μ
                if scf_data.screening_data.basis_function_screen_matrix[r, pp]
                    if start_index == 0 
                        start_index = r 
                    end
                    end_index = r
                end
                if start_index != 0 && (!scf_data.screening_data.basis_function_screen_matrix[r, pp] || r == scf_data.μ) 
                    push!(scf_data.screening_data.non_zero_ranges[pp], start_index:end_index)
                    range_length = end_index - start_index + 1
                    push!(scf_data.screening_data.non_zero_sparse_ranges[pp], non_zero_index:(non_zero_index + range_length - 1))
                    non_zero_index += range_length 
                    
                    start_index = 0
                    end_index = 0
                end
            end
        end
    end
    jc_timing.timings[JCTC.screening_time] = screening_time
    jc_timing.timings[JCTC.screening_metadata_time] = screning_metadata_time
    jc_timing.timings[JCTC.screened_indices_count] = scf_data.screening_data.screened_indices_count

end