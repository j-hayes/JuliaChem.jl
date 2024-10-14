using HDF5
include("../src/shared/JCTiming.jl")

#docs 
# Take the data from JCtiming and save it to an hdf5 file which does not need 
# The rest of JuliaChem to be read, only the JCTC keys 
function save_jc_timings_to_hdf5(jc_timing::JCTiming, file_path::String)
    # write stuff struct to hdf5 
    h5open("$(file_path)", "w") do file
        # write_dictionary_to_hdf5{String}(file, jc_timing.non_timing_data, "non_timing_data")
        save_run_level_data(file, jc_timing) #run level data
        # iteration times
        # fock build times 


    end
end


# function write_dictionary_to_hdf5{T}(file, dictionary::Dict{String, T}, file_name) where T
#     dict_keys = collect(keys(dictionary))
#     dict_values = collect(values(Dict))
#     write_arr = Array{String}(undef, length(dict_keys),2)
#     for i in eachindex(dict_keys)
#         write_arr[i, 1] = dict_keys[i]
#         write_arr[i, 2] = dict_values[i]
#     end
#     write(file, file_name, write_arr)
# end



function check_value_exists(key::String, dict::Dict{String, String}) :: String
    if haskey(dict, key)
        return dict[key]
    else
        return ""
    end
end

function save_run_level_data(file, jc_timing::JCTiming)
    str_array = Array{String}(undef, 11, 2)


    str_array[1, 1] = JCTC.run_name
    str_array[1, 2] = jc_timing.run_name

    str_array[2, 1] = JCTC.run_time
    str_array[2, 2] = string(jc_timing.run_time)

    str_array[3, 1] = JCTC.converged
    str_array[3, 2] = string(jc_timing.converged)

    str_array[4, 1] = JCTC.scf_energy
    str_array[4, 2] = string(jc_timing.scf_energy)

    str_array[5, 1] = JCTC.n_basis_functions
    str_array[5, 2] = check_value_exists(JCTC.n_basis_functions, jc_timing.non_timing_data)

    str_array[6, 1] = JCTC.n_auxiliary_basis_functions
    str_array[6, 2] = check_value_exists(JCTC.n_auxiliary_basis_functions, jc_timing.non_timing_data)

    str_array[7, 1] = JCTC.n_electrons
    str_array[7, 2] = check_value_exists(JCTC.n_electrons, jc_timing.non_timing_data)

    str_array[8, 1] = JCTC.n_occupied_orbitals
    str_array[8, 2] = check_value_exists(JCTC.n_occupied_orbitals, jc_timing.non_timing_data)

    str_array[9, 1] = JCTC.n_atoms
    str_array[9, 2] = check_value_exists(JCTC.n_atoms, jc_timing.non_timing_data)

    str_array[10, 1] = JCTC.n_threads
    str_array[10, 2] = check_value_exists(JCTC.n_threads, jc_timing.non_timing_data)

    str_array[11, 1] = JCTC.n_ranks
    str_array[11, 2] = check_value_exists(JCTC.n_ranks, jc_timing.non_timing_data)

    write(file, JCTC.run_level_data , str_array)
end


using Test
function check_run_level_data(data::Array{String, 2})

    #check the keys are correct
    @test data[1, 1] == JCTC.run_name
    @test data[2, 1] == JCTC.run_time
    @test data[3, 1] == JCTC.converged
    @test data[4, 1] == JCTC.scf_energy
    @test data[5, 1] == JCTC.n_basis_functions
    @test data[6, 1] == JCTC.n_auxiliary_basis_functions
    @test data[7, 1] == JCTC.n_electrons
    @test data[8, 1] == JCTC.n_occupied_orbitals
    @test data[9, 1] == JCTC.n_atoms
    @test data[10, 1] == JCTC.n_threads
    @test data[11, 1] == JCTC.n_ranks

    #check the values are correct 
    @test data[1, 2] == "test"
    @test data[2, 2] == "1.0"
    @test data[3, 2] == "true"
    @test data[4, 2] == "-1.0"
    @test data[5, 2] == "100"
    @test data[6, 2] == "1000"
    @test data[7, 2] == "10"
    @test data[8, 2] == "5"
    @test data[9, 2] == "4"
    @test data[10, 2] == "5"
    @test data[11, 2] == "1"
end

function test()
    jc_timing = create_jctiming()

    jc_timing.run_time = 1.0
    jc_timing.run_name = "test"
    jc_timing.converged = true
    jc_timing.scf_energy = -1.0
    jc_timing.non_timing_data[JCTC.n_basis_functions] = "100"
    jc_timing.non_timing_data[JCTC.n_auxiliary_basis_functions] = "1000"
    jc_timing.non_timing_data[JCTC.n_electrons] = "10"
    jc_timing.non_timing_data[JCTC.n_occupied_orbitals] = "5"
    jc_timing.non_timing_data[JCTC.n_atoms] = "4"
    jc_timing.non_timing_data[JCTC.n_threads] = "5"
    jc_timing.non_timing_data[JCTC.n_ranks] = "1"

    jc_timing.non_timing_data[JCTC.screened_indices_count] = "10000"
    jc_timing.non_timing_data[JCTC.unscreened_exchange_blocks] = "10"
    jc_timing.non_timing_data[JCTC.iteration_time] = "1.0"

    
    jc_timing.non_timing_data["GPU_num_devices"] = "1"
    jc_timing.non_timing_data["GPU_data_size_MB"] = "1000"
    

    save_jc_timings_to_hdf5(jc_timing, "/pscratch/sd/j/jhayes1/source/JuliaChem.jl/testoutputs/test_jc_timings.h5")

    # read in the data
    h5open("/pscratch/sd/j/jhayes1/source/JuliaChem.jl/testoutputs/test_jc_timings.h5", "r") do file
        data = read(file, JCTC.run_level_data)
        check_run_level_data(data)
    end

end

test()
