include("../tools/travis/travis-rhf.jl")
using Serialization
function run_s22_test(s22_number, output_file_path, number_of_samples )
    #== select input files ==#
    S22_directory = joinpath(@__DIR__, "../example_inputs/S22/")
    inputs = readdir(S22_directory)
    inputs .= S22_directory .* inputs

    input = inputs[s22_number]
    
    JuliaChem.initialize()
    
    run_test(input, output_file_path, number_of_samples)

    JuliaChem.finalize()
end

function run_test(input_file_path:: String, results_file_output::String, number_of_samples::Int)
    println("Running RHF vs DF-RHF vs DF guess RHF test: $input_file_path")
    # run to compile ignore timings for these
    test_results = Dict()
    RHF = "RHF"
    DF_RHF = "DF-RHF"
    DF_Guess_RHF = "DF-Guess-RHF"
    test_results[RHF] = Dict()
    test_results[DF_RHF] = Dict()
    test_results[DF_Guess_RHF] = Dict()
    
    rhf_results = travis_rhf(input_file_path)
    df_results = travis_rhf_density_fitting(input_file_path, "cc-pVTZ-JKFIT", true)
    df_guess_results = travis_rhf_density_fitting(input_file_path, "cc-pVTZ-JKFIT", false)

    for iter in 1:number_of_samples
        println("Running RHF test: $iter")
        rhf_results = travis_rhf(input_file_path)
        collect_run_results(test_results, RHF, rhf_results, iter)

        println("Running DF-RHF test: $iter")
        df_results = travis_rhf_density_fitting(input_file_path, "cc-pVTZ-JKFIT", false)
        collect_run_results(test_results, DF_RHF, df_results, iter)

        println("Running RHF with DF guess test: $iter")
        df_guess_results = travis_rhf_density_fitting(input_file_path, "cc-pVTZ-JKFIT", true)
        collect_run_results(test_results, DF_Guess_RHF, df_guess_results, iter)

    end

    serialize(results_file_output, test_results)
end

function collect_run_results(results, run_name, rhf_results, iter)
    results_dict = Dict()

    results_dict["Energy"] = rhf_results.Energy["Energy"]
    results_dict["Iteration_Times"] = rhf_results.Energy["Timings"].iteration_times
    results_dict["Run_Time"] = rhf_results.Energy["Timings"].run_time
    results_dict["density_fitting_iteration_range_start"] = rhf_results.Energy["Timings"].density_fitting_iteration_range[1]
    results_dict["density_fitting_iteration_range_end"] = rhf_results.Energy["Timings"].density_fitting_iteration_range[end]
    results_dict["Properties"] = rhf_results.Properties
    results[run_name][iter] = results_dict
   
    
end



args_length = length(ARGS)
if args_length == 0
    s22_number = 10
    output_file_path = "./S22_results_10.txt"
    number_of_samples = 2
end
if args_length >= 1
    s22_number = ARGS[1]
end
if args_length >= 2
    output_file_path = ARGS[2]
end
if args_length >= 3
    number_of_samples = parse(Int, ARGS[3])
end

run_s22_test(s22_number, output_file_path, number_of_samples)

