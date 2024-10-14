
# Julia Chem Timing Constants 
module JCTC


    #file names when saving to hdf5 or other key value store 
    const run_level_data = "run_level_data"
    const scf_options = "scf_options"
    const timings = "timings"
    const non_timing_data = "non_timing_data"
    #...


    const run_time = "run_time"
    const run_name = "run_name"
    const converged = "converged" 
    const scf_energy = "scf_energy"


    const n_basis_functions = "n_basis_functions"
    const n_electrons = "n_electrons"
    const n_occupied_orbitals = "n_oo"
    const n_atoms = "n_atoms"

    const n_threads = "n_threads" 
    const n_ranks = "n_ranks"
    const n_iterations = "n_iterations"
    const size_scf_data_MB = "size_scf_data_MB"


    #options constants 
    const JCTC.density_fitting = "density_fitting"
    const JCTC.contraction_mode = "contraction_mode"
    const JCTC.load = "load"
    const JCTC.guess = "guess"
    const JCTC.energy_convergence = "energy_convergence"
    const JCTC.density_convergence = "density_convergence"
    const JCTC.df_energy_convergence = "df_energy_convergence"
    const JCTC.df_density_convergence = "df_density_convergence"
    const JCTC.max_iterations = "max_iterations"
    const JCTC.df_max_iterations = "df_max_iterations"
    const JCTC.df_exchange_block_width = "df_exchange_block_width"
    const JCTC.df_screening_sigma = "df_screening_sigma"
    const JCTC.df_screen_exchange = "df_screen_exchange"

    #iteration times 
    const iteration_time = "iteration_time-"#
    const fock_time = "fock_time-"#


    #fock build times
    const form_J_AB_inv_time = "form_J_AB_inv_time" 
    const density_time = "density_time-" # CPU D  #GPU D S
    const H_time = "H_time"# CPU D S #GPU C S 
    const H_add_time = "H_add_time-"# CPU D S  #GPU
    const K_time = "K_time-"
    const W_time = "W_time-"
    const J_time = "J_time-"
    const V_time = "V_time-"
    const screening_time = "screening_time"# CPU X S  #GPU
    const fock_MPI_time = "fock_MPI_time-"# ALL 
    const screening_metadata_time = "screening_metadata_time"# CPU X S  #GPU X S 
    #




    #DF specific
    const contraction_algorithm = "contraction_algorithm"
    const B_time = "B_time" 

    const n_auxiliary_basis_functions = "n_auxiliary_basis_functions"
    const three_eri_time = "three_eri_time"
    const two_eri_time = "two_eri_time"
    const screened_indices_count = "screened_indices_count" 
    const unscreened_exchange_blocks = "unscreened_exchange_blocks" 
    const total_exchange_blocks = "total_exchange_blocks" 

    #DF Guess specific
    const DF_iteration_range_start = "DF_iteration_range_start"
    const DF_iteration_range_end = "DF_iteration_range_end"

    #GPU specific
    const fock_gpu_cpu_copy_time = "fock_gpu_cpu_copy_time-"
    const gpu_copy_J_time = "gpu_copy_J_time-"
    const gpu_copy_sym_time = "gpu_copy_sym_time-"
    const total_fock_gpu_time = "total_fock_gpu_time-" 
    const GPU_K_time = "GPU_-N-_K_time-" 
    const GPU_W_time = "GPU_-N-_W_time-" 
    const GPU_J_time = "GPU_-N-_J_time-" 
    const GPU_V_time = "GPU_-N-_V_time-" 
    const gpu_fock_time = "GPU_-N-_fock_time-"


    const GPU_H_add_time = "GPU_-N-_H_add_time-"
    const GPU_non_zero_coeff_time = "GPU_-N-_non_zero_coeff_time-" 
    const GPU_density_time = "GPU_-N-_density_time-"
    const GPU_screening_setup_time = "GPU_screening_setup_time" 
    const GPU_num_devices = "GPU_num_devices"
    const GPU_data_size_MB = "GPU_-N-_data_size_MB" 
end

mutable struct JCTiming
    run_time :: Float64
    run_name :: String
    converged :: Bool
    scf_energy :: Float64
    non_timing_data :: Dict{String, String}
    options :: Dict{String, String}
    timings :: Dict{String, Float64} # keys should be prefixed name_of_timing- then the iteration number
end

function create_jctiming()
    return JCTiming(
        0.0, # run_time
        "", # run_name
        false, # converged
        0.0, # scf_energy
        Dict{String, String}(), # non_timing_data
        Dict{String, String}(), # options
        Dict{String, Float64}() # timings
        )
end

@inline function JCTiming_key(key::String, iteration::Int)
    return key * string(iteration)
end

@inline function JCTiming_GPUkey(key::String, device::Int, iteration::Int)
    return JCTiming_GPUkey(key, device) * string(iteration)
end

@inline function JCTiming_GPUkey(key::String, device::Int)
    return replace(key, "-N-" => string(device))
end
# function print_iteration_times(timings::JCTiming)
#     #print integer keys first, then the rest
#     timings_deep_copy = deepcopy(timings)
#     sorted_pairs = sort(collect(pairs(timings_deep_copy.iteration_times)), by= x->get_sort_index_from_key(x[1]))
#     for key_value_pair in sorted_pairs
#         if isnothing(tryparse(Int, key_value_pair[1]))
#             continue
#         end
#         println("$(key_value_pair[1]): $(key_value_pair[2]) seconds")
#         delete!(timings_deep_copy.iteration_times, key_value_pair[1])    
#     end
#     sorted_pairs = sort(collect(pairs(timings_deep_copy.iteration_times)), by= x->x[1])
#     for key_value_pair in sorted_pairs
#         println("$(key_value_pair[1]): $(key_value_pair[2]) seconds")
#     end 
# end

# function get_sort_index_from_key(key::String)
#    parsed_key = tryparse(Int, key)
#    if isnothing(parsed_key)
#      return -1
#    else
#      return parsed_key
#    end
# end




export JCTiming, create_jctiming, JCTC, JCTiming_key,JCTiming_GPUkey



