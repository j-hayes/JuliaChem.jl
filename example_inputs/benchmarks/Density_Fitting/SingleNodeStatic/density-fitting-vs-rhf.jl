include("../benchmark.jl")
using Serialization

function run_s22_test_BLAS_VS_TensorOp(inputs_dir, s22_number, output_file_path, number_of_samples)
    #== select input files ==#
    S22_directory = joinpath(@__DIR__, inputs_dir)
    inputs = readdir(S22_directory)
    inputs .= S22_directory .* inputs

    input = inputs[s22_number]
    

    run_test(input, output_file_path, number_of_samples, s22_number, "6-311++G(2d,2p)", "aug-cc-pVTZ-JKFIT")

end

function run_test(input_file_path:: String, results_file_output::String, number_of_samples::Int, s22_number, basis, aux_basis)
    println("Running RHF vs DF-RHF vs DF guess RHF test: $input_file_path")
    # run to compile ignore timings for these
    test_results = Dict()
    
    DF_RHF_BLAS = "DF_RHF_HCORE"
    DF_RHF_TENOP = "DF_GUESS_RHF_HCORE_TENOP"
    RHF_HCORE = "RHF_HCORE"

    test_results[DF_RHF_BLAS] = Dict()
    test_results[DF_RHF_TENOP] = Dict()
    test_results[RHF_HCORE] = Dict()

    guess = "hcore"
    contraction_mode_TO = "TensorOperations"
    contraction_mode_BLAS = "BLAS"
    static_load = "static"
    
    small_compile_run_input_file = "/home/jackson/source/JuliaChem.jl/example_inputs/density_fitting/start_up.json"
    run_df_rhf(small_compile_run_input_file, basis ,aux_basis, true, guess, contraction_mode_TO, static_load)
    run_df_rhf(small_compile_run_input_file, basis ,aux_basis, true, guess, contraction_mode_BLAS, static_load)
    run_rhf(small_compile_run_input_file, basis, guess, static_load)


    println("doing s22 = $s22_number")
    for iter in 1:number_of_samples
        do_named_run_dfrhf(test_results, iter, DF_RHF_BLAS, input_file_path, basis, aux_basis, guess, false, contraction_mode_BLAS, static_load)
        do_named_run_dfrhf(test_results, iter, DF_RHF_TENOP, input_file_path, basis, aux_basis, guess, false, contraction_mode_TO, static_load)
        do_named_run_rhf(test_results, iter, RHF_HCORE, input_file_path, basis, guess, static_load)
    end

    serialize(results_file_output, test_results)
end

function do_named_run_rhf(test_results, iter, name, input_file_path, basis,  guess, load)
    println("Running $name test: $iter")
    results = run_rhf(input_file_path, basis, guess, load)
    collect_run_results(test_results, name, results, iter)
    flush(stdout)

end

function do_named_run_dfrhf(test_results, iter, name, input_file_path, basis, aux_basis ,guess, df_is_guess, contraction_mode, load)
    println("Running $name test: $iter")
    results = run_df_rhf(input_file_path, basis, aux_basis, df_is_guess, guess, contraction_mode, load)
    collect_run_results(test_results, name, results, iter)
    flush(stdout)
end

function collect_run_results(results, run_name, rhf_results, iter)
    results_dict = Dict()
    results_dict["Keywords"] = rhf_results.Keywords
    results_dict["Molecule"] = rhf_results.Molecule
    results_dict["Model"] = rhf_results.Model
    results_dict["Energy"] = rhf_results.Energy["Energy"]
    results_dict["Iteration_Times"] = rhf_results.Energy["Timings"].iteration_times
    results_dict["Run_Time"] = rhf_results.Energy["Timings"].run_time
    results_dict["density_fitting_iteration_range_start"] = rhf_results.Energy["Timings"].density_fitting_iteration_range[1]
    results_dict["density_fitting_iteration_range_end"] = rhf_results.Energy["Timings"].density_fitting_iteration_range[end]
    results_dict["Properties"] = rhf_results.Properties
    results[run_name][iter] = results_dict
   
    
end
number_of_samples = 2
inputs_dir = "../../../S22/"

JuliaChem.initialize()
for s22_number in 1:22
    output_file_path = "./results/S22_1NODE_20threads_results_$(s22_number)_static.data"
    run_s22_test_BLAS_VS_TensorOp(inputs_dir, s22_number, output_file_path, number_of_samples)
    results =  deserialize(output_file_path)
end
JuliaChem.finalize()

