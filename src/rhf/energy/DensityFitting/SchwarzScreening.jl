using HDF5
""" Schwarz screening for DF-RHF as described in Huang et al. 2020 https://doi.org/10.1063/1.5129452
if σ is a threshold on the size of (pq|P), 
then (pq|P) can be neglected if (pq∣pq) < σ^2 / maxP(P∣P) 
where max_P(P|P) is the maximum value of (P|P) over all P
Arguments
=========
σ = 10^(−12) is used in huang et. al 2020 screening parameter
"""
function schwarz_screen_itegrals_df(scf_data, σ, max_P_P, basis_sets, jeri_engine_thread)


    #hdf5 file for adding schwarz screening information
    #open the file and create if it doesn't exist 

    

    basis = basis_sets.primary
    basis_set_length = length(basis)
    shell_screen_matrix = zeros(Bool,basis_set_length, basis_set_length) # true means keep the shell pair, false means it is screened
    basis_function_screen_matrix = zeros(Bool,scf_data.μ, scf_data.μ) # true means keep the basis function pair, false means it is screened

    n_shell_indicies = basis_set_length * (basis_set_length + 1) ÷ 2 # # of triangular shell pairs (pq)
    max_am = max_ang_mom(basis) 
    batch_size = eri_quartet_batch_size(max_am)
    nthreads = Threads.nthreads()
    eri_quartet_batch_thread = [ Vector{Float64}(undef, batch_size) for thread in 1:nthreads ]

    debug_pqpq_dict = Dict{Int, Vector{Float64}}() # [index, (pq|pq) integrals]
    debug_pqpq_dict_upper_triangular = Dict{Int, Vector{Float64}}() # [index, (pq|pq) integrals]
    debug_index_to_pq = Dict{Int64, Tuple{Int, Int}}() # [index, (ish,jsh)]
    debug_index_to_pq_upper_triangular = Dict{Int64, Tuple{Int, Int}}() # [index, (ish,jsh)]
    shell_info = Dict{Int, Tuple{Int, Int, Int}}() # [shell_index, (n_bas, am, pos)]

    for (i, shell) in enumerate(basis)
        shell_info[i] = (shell.nbas, shell.am, shell.pos - 1)
    end

    println("sigma squared: ", σ^2)
    println("max_P_P: ", max_P_P)
    println("threshold for screening: ", (σ^2) / max_P_P)

    threshold = (σ^2) / max_P_P #10.0^-10 hardcode for comparison with EXESS GPU DF-RHF
    for thread in 1:nthreads
        begin
            for index in thread:nthreads:n_shell_indicies
                bra_pair = index
                ket_pair = index
                ish = decompose(index)
                jsh = index - triangular_index(ish)

                upper_t_index = upper_triangular_index(jsh-1, ish-1, basis_set_length)

                debug_index_to_pq[index] = (ish, jsh)
                debug_index_to_pq_upper_triangular[upper_t_index] = (jsh-1, ish-1)
                μ_shell = basis[ish]
                ν_shell = basis[jsh]
            
                nμ = μ_shell.nbas
                nν = ν_shell.nbas

                μ_position = μ_shell.pos
                ν_position = ν_shell.pos



                eri_quartet_batch_thread[thread] .= 0.0
                JERI.compute_eri_block(jeri_engine_thread[thread], eri_quartet_batch_thread[thread],
                    ish, jsh, ish, jsh, bra_pair, ket_pair, nμ * nν, nμ * nν)

                axial_normalization_factor(eri_quartet_batch_thread[thread], μ_shell, ν_shell, μ_shell, ν_shell, nμ, nν, nμ, nν)
                debug_pqpq_dict[index] = copy(eri_quartet_batch_thread[thread])
                debug_pqpq_dict_upper_triangular[upper_t_index] = copy(eri_quartet_batch_thread[thread])

                shell_pair_contracted = sum(eri_quartet_batch_thread[thread])

                if ish == 1 && jsh == 37 || ish == 37 && jsh == 1
                    println("shell pair (ish, jsh): ($ish, $jsh), shell_pair_contracted: $shell_pair_contracted")
                end  

                shell_screen_matrix[ish, jsh] = !(Base.abs_float(shell_pair_contracted) < threshold)
                shell_screen_matrix[jsh, ish] = shell_screen_matrix[ish, jsh]

                if shell_screen_matrix[ish, jsh] == false # if the shell pair is screened, then screen all the basis function pairs
                    for μμ::Int64 in μ_position:(μ_position+nμ-1)
                        for νν::Int64 in ν_position:(ν_position+nν-1)
                            basis_function_screen_matrix[μμ, νν] = false
                            basis_function_screen_matrix[νν, μμ] = false
                        end
                    end
                else #screen individual basis functions pairs within the shell pair
                    for μμ::Int64 in μ_position:(μ_position+nμ-1)
                        for νν::Int64 in ν_position:(ν_position+nν-1)
                            μνμν = 1 + (νν - ν_position) + nν * (μμ - μ_position) + nν * nμ * (νν - ν_position) + nν * nμ * nν * (μμ - μ_position)
                            eri = eri_quartet_batch_thread[thread][μνμν]
                            basis_function_screen_matrix[μμ, νν] = !(Base.abs_float(eri) < threshold)
                            basis_function_screen_matrix[νν, μμ] = basis_function_screen_matrix[μμ, νν]
                        end
                    end
                end
            end
        end # end of thread spawn
    end # end of thread sync 
    sparse_pq_index_map = zeros(Int64, scf_data.μ, scf_data.μ)
    sparse_index = 1
    for pp::Int64 in 1:scf_data.μ
        for qq::Int64 in 1:scf_data.μ
            if basis_function_screen_matrix[qq,pp] == true
                sparse_pq_index_map[qq,pp] = sparse_index
                sparse_index += 1                
            end
        end
    end

  #sort the dictionaries and put keys and values in arrays to go into hdf5
#   debug_pqpq_keys = Vector{Int64}(undef, length(debug_pqpq_dict))
#   debug_pqpq_values = Matrix{Float64}(undef, length(first(values(debug_pqpq_dict))), length(debug_pqpq_dict))
  
#   for (i, key) in enumerate(sort(collect(keys(debug_pqpq_dict))))
#       debug_pqpq_keys[i] = key
#       debug_pqpq_values[:, i] .= debug_pqpq_dict[key]
#   end

#   debug_pqpq_keys_upper_triangular = Vector{Int64}(undef, length(debug_pqpq_dict_upper_triangular))
#   debug_pqpq_values_upper_triangular = Matrix{Float64}(undef, length(first(values(debug_pqpq_dict_upper_triangular))), length(debug_pqpq_dict_upper_triangular))

#   for (i, key) in enumerate(sort(collect(keys(debug_pqpq_dict_upper_triangular))))
#       debug_pqpq_keys_upper_triangular[i] = key
#       debug_pqpq_values_upper_triangular[:, i] .= debug_pqpq_dict_upper_triangular[key]
#   end

#   debug_index_to_pq_keys = Vector{Int64}(undef, length(debug_index_to_pq))
#   debug_index_to_pq_values = Matrix{Int64}(undef, 2, length(debug_index_to_pq))
#   for (i, key) in enumerate(sort(collect(keys(debug_index_to_pq))))
#       debug_index_to_pq_keys[i] = key
#       debug_index_to_pq_values[:, i] .= debug_index_to_pq[key] .- 1
#   end

#   debug_index_to_pq_keys_upper_triangular = Vector{Int64}(undef, length(debug_index_to_pq_upper_triangular))
#   debug_index_to_pq_values_upper_triangular = Matrix{Int64}(undef, 2, length(debug_index_to_pq_upper_triangular))
  
  #print the sorted upper triangle keys
#   println(sort(collect(keys(debug_index_to_pq_upper_triangular))))
#   for (i, key) in enumerate(sort(collect(keys(debug_index_to_pq_upper_triangular))))
#       debug_index_to_pq_keys_upper_triangular[i] = key
#       debug_index_to_pq_values_upper_triangular[:, i] .= debug_index_to_pq_upper_triangular[key]
#   end

#   shell_info_keys = Vector{Int64}(undef, length(shell_info))
#   shell_info_values = Array{Int,2}(undef, 3, length(shell_info))
#   for (i, key) in enumerate(sort(collect(keys(shell_info))))
#       shell_info_keys[i] = key
#       shell_info_values[:, i] .= shell_info[key]
#   end

#  unscreened_pq_count = count(!, basis_function_screen_matrix)
#  shell_screen_as_int = Int.(shell_screen_matrix)
#  basis_screen_as_int = Int.(basis_function_screen_matrix)
#  h5file = h5open("schwarz_screening_data-gly8.h5", "w")   
#    write(h5file, "shell_screen_matrix", shell_screen_as_int)
#    write(h5file, "basis_function_screen_matrix", basis_screen_as_int)
#    write(h5file, "sparse_pq_index_map", sparse_pq_index_map .-1 )
#    write(h5file, "debug_pqpq_keys", debug_pqpq_keys .-1)
#    write(h5file, "debug_pqpq_values", debug_pqpq_values)
#    write(h5file, "debug_pqpq_keys_upper_triangular", debug_pqpq_keys_upper_triangular)
#    write(h5file, "debug_pqpq_values_upper_triangular", debug_pqpq_values_upper_triangular)
#    write(h5file, "debug_index_to_pq_keys", debug_index_to_pq_keys .- 1)
#    write(h5file, "debug_index_to_pq_values",  debug_index_to_pq_values .- 1)
#    write(h5file, "debug_index_to_pq_keys_upper_triangular", debug_index_to_pq_keys_upper_triangular )
#    write(h5file, "debug_index_to_pq_values_upper_triangular", debug_index_to_pq_values_upper_triangular)
#    write(h5file, "unscreened_basis_count", [unscreened_pq_count])
#    write(h5file, "shell_info_keys",  shell_info_keys)
#    write(h5file, "shell_info_values", shell_info_values)
#    close(h5file)

    return shell_screen_matrix, basis_function_screen_matrix, sparse_pq_index_map
end


function get_max_P_P(two_center_integrals)
    P = size(two_center_integrals)[1]
    maxP = floatmin(Float64) # smallest possible float
    for p in 1:P
        if Base.abs_float(two_center_integrals[p,p]) > maxP
            maxP = abs(two_center_integrals[p,p])
        end
    end
    return maxP
end

function setup_unscreened_screening_matricies(basis_sets, scf_data)
    basis = basis_sets.primary
    basis_set_length = length(basis)
    scf_data.screening_data.shell_screen_matrix = ones(Bool,basis_set_length, basis_set_length) # true means keep the shell pair, false means it is screened
    scf_data.screening_data.basis_function_screen_matrix = ones(Bool,scf_data.μ, scf_data.μ) # true means keep the basis function pair, false means it is screened
    scf_data.screening_data.screened_indices_count = scf_data.μ^2
    scf_data.screening_data.sparse_pq_index_map = zeros(Int64, (scf_data.μ, scf_data.μ))

    Threads.@threads for pp in 1:scf_data.μ
      for qq in 1:scf_data.μ
        #2D index to 1D index row major (why is this row major?)
        scf_data.screening_data.sparse_pq_index_map[pp, qq] = pp + (qq - 1) * scf_data.μ
      end
    end
end

export schwarz_screen_itegrals_df, get_max_P_P, setup_unscreened_screening_matricies