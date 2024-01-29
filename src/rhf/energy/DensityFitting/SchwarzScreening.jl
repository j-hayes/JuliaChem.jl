""" Schwarz screening for DF-RHF as described in Huang et al. 2020 https://doi.org/10.1063/1.5129452
if σ is a threshold on the size of (pq|P), 
then (pq|P) can be neglected if (pq∣pq) < σ^2 / maxP(P∣P) 
where max_P(P|P) is the maximum value of (P|P) over all P
Arguments
=========
σ = 10^(−12) is used in huang et. al 2020 screening parameter
"""
function schwarz_screen_itegrals_df(scf_data, σ, max_P_P, basis_sets, jeri_engine_thread)

    basis = basis_sets.primary
    basis_set_length = length(basis)
    shell_screen_matrix = trues(basis_set_length, basis_set_length) # true means keep the shell pair, false means it is screened
    basis_function_screen_matrix = trues(scf_data.μ, scf_data.μ) # true means keep the basis function pair, false means it is screened

    n_shell_indicies = get_n_shell_indicies(basis_set_length)
    max_am = max_ang_mom(basis) 
    batch_size = eri_quartet_batch_size(max_am)
    nthreads = Threads.nthreads()
    eri_quartet_batch_thread = [ Vector{Float64}(undef, batch_size) for thread in 1:nthreads ]

    σ_squared = σ^2

    for index in 1:n_shell_indicies
        bra_pair, ket_pair, ish,jsh,ksh,lsh = decompose_shell_index_ijkl(index)
        # take only where (pq|pq) and p >= q 
        if ish != ksh || jsh != lsh || jsh > ish 
            continue
        end



        μ_shell = basis[ish] 
        ν_shell = basis[jsh] 
        λ_shell = basis[ksh] 
        σ_shell = basis[lsh]    # σ shell not to be confused with screening parameter σ   

        nμ = μ_shell.nbas
        nν = ν_shell.nbas
        nλ = λ_shell.nbas
        nσ = σ_shell.nbas

        μ_position = μ_shell.pos
        ν_position = ν_shell.pos

        eri_quartet_batch_thread[1] .= 0.0
        JERI.compute_eri_block(jeri_engine_thread[1], eri_quartet_batch_thread[1], 
        ish,jsh,ksh,lsh, bra_pair, ket_pair, nμ*nν, nλ*nσ)

        axial_normalization_factor(eri_quartet_batch_thread[1], μ_shell, ν_shell, λ_shell, σ_shell, nμ, nν, nλ, nσ)

        shell_pair_contracted = sum(abs.(eri_quartet_batch_thread[1]))

        shell_screen_matrix[ish,jsh] = !(shell_pair_contracted <  σ_squared / max_P_P)
        shell_screen_matrix[jsh,ish] = shell_screen_matrix[ish,jsh]

        if shell_screen_matrix[ish,jsh] == false # if the shell pair is screened, then screen all the basis function pairs
            for μμ::Int64 in μ_position:(μ_position+nμ-1) 
                for νν::Int64 in ν_position:(ν_position+nν-1) 
                    basis_function_screen_matrix[μμ,νν] = false
                    basis_function_screen_matrix[νν,μμ] = false
                end
            end
        else #screen individual basis functions pairs within the shell pair
            for μμ::Int64 in μ_position:(μ_position+nμ-1) 
                for νν::Int64 in ν_position:(ν_position+nν-1) 
                    μνμν = 1 + (νν-ν_position) + nν*(μμ-μ_position) + nν*nμ*(νν-ν_position) + nν*nμ*nν*(μμ-μ_position)
                    eri = eri_quartet_batch_thread[1][μνμν] 
                    basis_function_screen_matrix[μμ,νν] = !(abs(eri) <  σ_squared / max_P_P)
                    basis_function_screen_matrix[νν,μμ] = basis_function_screen_matrix[μμ,νν]
                end
            end
        end
    end

    sparse_pq_index_map = zeros(Int64, scf_data.μ, scf_data.μ)
    index = 1
    for μμ::Int64 in 1:scf_data.μ
        for νν::Int64 in 1:scf_data.μ
            if basis_function_screen_matrix[μμ,νν] == true
                sparse_pq_index_map[μμ,νν] = index
                index += 1                
            end
        end
    end
    return shell_screen_matrix, basis_function_screen_matrix, sparse_pq_index_map
end


function get_max_P_P(two_center_integrals)
    P = size(two_center_integrals)[1]
    maxP = floatmin(Float64) # smallest possible float
    for p in 1:P
        if abs(two_center_integrals[p,p]) > maxP
            maxP = two_center_integrals[p,p]
        end
    end
    return maxP
end

export schwarz_screen_itegrals_df, get_max_P_P