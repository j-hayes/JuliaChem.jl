using Serialization
function get_results()
    
end

function calculate_energy_ΔE(results, run_name1, run_name2)
    run_name1_energy = results[run_name1][1]["Energy"]
    run_name2_energy = results[run_name2][1]["Energy"]
    return run_name1_energy - run_name2_energy
end
function main(results_data_file_path)

    RHF = "RHF"
    DF_RHF = "DF-RHF"
    DF_Guess_RHF = "DF-Guess-RHF"
    
    results = deserialize(results_data_file_path)
    ΔE_DF_RHF = calculate_energy_ΔE(results, RHF, DF_RHF)
    ΔE_DF_Guess_RHF = calculate_energy_ΔE(results, RHF, DF_Guess_RHF)

    println("ΔE RHF vs DF-RHF: $ΔE_DF_RHF")
    println("ΔE DF-Guess-RHF: $ΔE_DF_Guess_RHF")

    RHF_DF_RHF_Speedup = calculate_speedup(results, RHF, DF_RHF)
    RHF_DF_Guess_RHF_Speedup = calculate_speedup(results, RHF, DF_Guess_RHF)

    println("RHF vs DF-RHF Speedup: $RHF_DF_RHF_Speedup")
    println("RHF vs DF-Guess-RHF Speedup: $RHF_DF_Guess_RHF_Speedup")

end

function calculate_speedup(results, run_name1, run_name2)
    
    total_runtime_1 = 0.0
    total_runtime_2 = 0.0
    iteration_keys = collect(keys(results[run_name1]))
    for iteration in iteration_keys
        total_runtime_1 += results[run_name1][iteration]["Run_Time"]
        total_runtime_2 += results[run_name2][iteration]["Run_Time"]
    end
    number_of_iterations = length(iteration_keys)
    average_runtime_1 = total_runtime_1 / number_of_iterations
    average_runtime_2 = total_runtime_2 / number_of_iterations
    
    speedup = average_runtime_1 / average_runtime_2
    return speedup
end


main("/home/jackson/source/JuliaChem.jl/S22_results_10.txt")