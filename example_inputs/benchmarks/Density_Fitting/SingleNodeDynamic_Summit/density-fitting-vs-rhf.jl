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
    
    DF_RHF_BLAS = "DF_RHF_BLAS"
    DF_RHF_TENOP = "DF_RHF_TENOP"
    DF_RHF_GPU = "DF_RHF_GPU"
    RHF_HCORE = "RHF_HCORE"
    RHF_DF_GUESS = "RHF_DF_GUESS"

    
    println("doing s22 = $s22_number")
    #throw away runs 

    guess = "hcore"
    contraction_mode_TO = "TensorOperations"
    contraction_mode_BLAS = "BLAS"
    contraction_mode_GPU = "GPU"
    loads = ["static", "dynamic"]
    start_up_basis = "6-31G"
    start_up_aux_basis = "cc-pVTZ-JKFIT"
    small_compile_run_input_file = "../../../density_fitting/start_up.json"
    small_compile_run_input_file_df = "../../../density_fitting/start_up_df.json"
    
    
    for iter in 1:number_of_samples
        for load in loads
            
        #RHF
        run_rhf(small_compile_run_input_file, start_up_basis, guess, load)
        do_named_run_rhf(test_results, iter, "$(RHF_HCORE)_$(load)", input_file_path, basis, guess, load)

        #DF Guess
        run_df_rhf(small_compile_run_input_file, start_up_basis ,start_up_aux_basis, true, guess, contraction_mode_TO, load)
        do_named_run_dfrhf(test_results, iter, "$(RHF_DF_GUESS)_$(load)", input_file_path, basis, aux_basis, guess, true, contraction_mode_TO, load)

        #TensorOperations
        run_df_rhf(small_compile_run_input_file, start_up_basis ,start_up_aux_basis, false, guess, contraction_mode_TO, load)
        do_named_run_dfrhf(test_results, iter, "$(DF_RHF_TENOP)_$(load)", input_file_path, basis, aux_basis, guess, false, contraction_mode_TO, load)

        #BLAS
        run_df_rhf(small_compile_run_input_file, start_up_basis ,start_up_aux_basis, false, guess, contraction_mode_BLAS, load)
        do_named_run_dfrhf(test_results, iter, "$(DF_RHF_BLAS)_$(load)", input_file_path, basis, aux_basis, guess, false, contraction_mode_BLAS, load)
            
        #GPU
        run_df_rhf(small_compile_run_input_file, start_up_basis ,start_up_aux_basis, false, guess, contraction_mode_GPU, load)
        do_named_run_dfrhf(test_results, iter, "$(DF_RHF_GPU)_$(load)", input_file_path, basis, aux_basis, guess, false, contraction_mode_GPU, load)

        end
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

    if typeof(rhf_results.Basis) == JuliaChem.JCBasis.CalculationBasisSets
        results_dict["Basis_Count"] =  rhf_results.Basis.primary.norb
        results_dict["Aux_Basis_Count"] = rhf_results.Basis.auxillary.norb
    else
        results_dict["Basis_Count"] = rhf_results.Basis.norb
    end
    if !haskey(results, run_name)
        results[run_name] = Dict()
    end
    results[run_name][iter] = results_dict
   
    
end
function run_df_vs_rhf_test()
    println("running density fitting vs rhf")
    number_of_samples = 2
    inputs_dir = "../../../S22/"
    JuliaChem.initialize()
    
    println("=======================================")
    comm = MPI.COMM_WORLD
    if MPI.Comm_rank(comm) == 0
      println(" ")
      println("Number of worker processes: ", MPI.Comm_size(comm))
      println("Number of threads per process: ", Threads.nthreads())
      println("Number of threads in total: ",
      MPI.Comm_size(comm)*Threads.nthreads())
    end
    println("=======================================")


    for s22_number in 1:22
        output_file_path = "./results/summit_results_$(s22_number)_dynamic_21core_21threads.data"
        run_s22_test_BLAS_VS_TensorOp(inputs_dir, s22_number, output_file_path, number_of_samples)
    end

   

    JuliaChem.finalize()
end
