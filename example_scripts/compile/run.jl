# using MKL
using BLISBLAS
println("starting run.jl");flush(stdout)
using JuliaChem
println("done loading JuliaChem");flush(stdout)
using Dates
using Pkg
using LinearAlgebra
using MPI 
using Base.Threads
# using CUDA 

include("./Scripts/input_output_file_helper.jl")
include("./Settings/run_settings.jl")
include("./Settings/get_basis_info.jl")
include("./Scripts/rank_print.jl")
include("../jc_timings_write.jl") # JuliaChem/example_scripts/jc_timings_write.jl
include("./Scripts/replace_keywords.jl")

function main(ARGS)
    
    path_to_inputs = ARGS[1]
    path_to_outputs = ARGS[2]
    settings_id = ARGS[3]
    basis_id = ARGS[4]
    run_index_start = parse(Int, ARGS[5])
    number_of_runs = parse(Int, ARGS[6])



    input_files_paths, input_file_names = get_file_pathsV1(path_to_inputs)

    if length(ARGS) >= 8
        file_start_index = parse(Int, ARGS[7])
        file_end_index = parse(Int, ARGS[8])
        input_files_paths = input_files_paths[file_start_index:file_end_index]
        input_file_names = input_file_names[file_start_index:file_end_index]
    elseif length(ARGS) == 7
        file_start_index = parse(Int, ARGS[7])
        input_files_paths = input_files_paths[file_start_index:end]
        input_file_names = input_file_names[file_start_index:end]
    end


    scf_keywords = get_settings(settings_id)

    if haskey(ENV, "df_exchange_n_blocks")
        scf_keywords["df_exchange_n_blocks"] = parse(Int, ENV["df_exchange_n_blocks"])
    end
    basis, aux_basis = get_basis_names(basis_id)

    JuliaChem.initialize()
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    if rank == 0
        Pkg.status()
    end


    rank_println("starting run starting at index $(run_index_start)", rank)
    rank_println("Date of run: $(Dates.now())", rank)
    rank_println("settings_id: $settings_id", rank)
    rank_println("basis_id: $basis_id", rank)
    rank_println("basis: $basis", rank)
    rank_println("aux_basis: $aux_basis", rank)
    rank_println("BLAS config $(BLAS.get_config())", rank)

    if haskey(ENV, "df_adaptive_basis_limit")
        scf_keywords["df_adaptive_basis_limit"] = parse(Int, ENV["df_adaptive_basis_limit"])
        rank_println("df_adaptive_basis_limit: $(scf_keywords["df_adaptive_basis_limit"])", rank)
    end

    # scf_keywords["df_adaptive_basis_limit"] = 100

    print("scf_keywords: ")
    rank_display(scf_keywords, rank)


    rank_println("input_files_paths", rank)
    rank_println(input_files_paths, rank)
    rank_println("output file path", rank)
    rank_println(path_to_outputs, rank)


    
    rank_println("initialized JuliaChem", rank)
    flush(stdout)
    n_threads = Threads.nthreads()
    rank_println("Threads $n_threads", rank)

    if haskey(ENV, "OPENBLAS_NUM_THREADS") &&  ENV["OPENBLAS_NUM_THREADS"] != nothing
        rank_println("OPENBLAS_NUM_THREADS: $(ENV["OPENBLAS_NUM_THREADS"])", rank)
        open_blas_threads = parse(Int, ENV["OPENBLAS_NUM_THREADS"])
        BLAS.set_num_threads(open_blas_threads)
    end


    rank_println("BLAS Threads $(BLAS.get_num_threads())", rank)
    # BLAS.set_num_threads(n_threads)
    # println("done pinning threads")
    # create output directory if it does not exist
    if !isdir(path_to_outputs)
        mkdir(path_to_outputs)
    end

    scf_keywords["num_devices"] = 1

    output_print_level = 2
    for i in eachindex(input_files_paths)
        #create folder for path_to_outputs/input_file_names[i]
        input_file_path = input_files_paths[i]
        input_file_name = input_file_names[i]
        output_dir = create_output_folderV2(path_to_outputs, input_file_path, rank)
        run_file(input_file_path, input_file_name, output_dir, scf_keywords, basis, aux_basis, run_index_start, number_of_runs, output_print_level)
    end

    # JuliaChem.finalize()

end


function run_file(input_file_path, input_file_name, output_dir, scf_keywords, basis, aux_basis, run_index_start,number_of_runs, output_print_level)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    rank_println("running input file $input_file_path", rank)
  
    molecule, driver, model, keywords = JuliaChem.JCInput.run(input_file_path; output=output_print_level)
    model_replace_dict = Dict{String,Any}("basis" => basis, "auxiliary_basis" => aux_basis)
    
    replace_model(model, model_replace_dict)
    replace_scf_keywords(keywords, scf_keywords)

    for run_id in run_index_start:run_index_start+number_of_runs-1
        name = split(input_file_name, ".")[1]
        name = "$(name)_run_$run_id"
        rank_println("Running $name", rank)
        # try

            mol, basis = JuliaChem.JCBasis.run(molecule, model; output=output_print_level)
            JuliaChem.JCMolecule.run(mol)

            run_time = @elapsed rhf_energy = JuliaChem.JCRHF.Energy.run(mol, basis, keywords["scf"]; output=output_print_level)
            timings = rhf_energy["Timings"]
            timings.run_name = name
            timings.run_time = run_time

            save_jc_timings_to_hdf5(timings, joinpath(output_dir, "$(name)-$(rank).h5"))

            rhf_energy = nothing 

        # catch e
        #     rank_println("Error running file: $input_file_path, run $run_id", 0)
        #     rank_println("Error: $e", 0)
        # end
        # GC.gc(true)
        # CUDA.reclaim()
        # #cuda memory status 
        # display(CUDA.pool_status())
        # GC.gc(true)
        # CUDA.reclaim()
        # display(CUDA.pool_status())
    end
end

# main(ARGS)

inputs_path=ENV["inputs_path"]
outputs_path=ENV["outputs_path"]

println("using inputs from $inputs_path")
println("writing outputs to $outputs_path")

# screened
inputs = []
push!(inputs,inputs_path)
push!(inputs,"$outputs_path/DF_RHF_screenedCPU")
push!(inputs,"DF_RHF_screenedCPU")
push!(inputs,"cc-pvdz-ri")
push!(inputs,"1")
push!(inputs,"1")
push!(inputs,"5")
push!(inputs,"5")

main(inputs)

inputs = []
#dense
push!(inputs,inputs_path)
push!(inputs,"$outputs_path/DF_RHF_denseCPU")
push!(inputs,"DF_RHF_denseCPU")
push!(inputs,"cc-pvdz-ri")
push!(inputs,"1")
push!(inputs,"1")
push!(inputs,"5")
push!(inputs,"5")
main(inputs)

#do mixed precision 
inputs[2] = "$outputs_path/DF_RHF_mixedCPU"
ENV["DO_MIXED"] = "true"
main(inputs)

#RHF static CPU 
inputs = []

push!(inputs,inputs_path)
push!(inputs,"$outputs_path/RHF_staticCPU")
push!(inputs,"RHF_staticCPU")
push!(inputs,"cc-pvdz-ri")
push!(inputs,"1")
push!(inputs,"1")
push!(inputs,"5")
push!(inputs,"5")


# main(inputs)
# inputs = []

# push!(inputs,inputs_path)
# push!(inputs,"$outputs_path/DF_RHF_GPU_adaptive")
# push!(inputs,"DF_RHF_GPU_adaptive")
# push!(inputs,"cc-pvdz-ri")
# push!(inputs,"1")
# push!(inputs,"1")
# push!(inputs,"10")
# push!(inputs,"10")


# main(inputs)

# RHF_dynamicCPU
inputs = []
push!(inputs,inputs_path)
push!(inputs,"$outputs_path/RHF_dynamicCPU")
push!(inputs,"RHF_dynamicCPU")
push!(inputs,"cc-pvdz-ri")
push!(inputs,"1")
push!(inputs,"1")
push!(inputs,"5")
push!(inputs,"5")
push!(inputs, ARGS)

main(inputs)

# #Guess DF
# inputs = []
# push!(inputs,inputs_path)
# push!(inputs,"$outputs_path/DF_RHF_screenedCPU_as_guess")
# push!(inputs,"DF_RHF_screenedCPU_as_guess")
# push!(inputs,"cc-pvdz-ri")
# push!(inputs,"1")
# push!(inputs,"1")
# push!(inputs,"5")
# push!(inputs,"5")
# push!(inputs, ARGS)

# main(inputs)


# JuliaChem.finalize()


println("done with precompile JuliaChem Runs")
return 0