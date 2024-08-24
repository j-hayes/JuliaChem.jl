using Serialization
using JuliaChem
using JuliaChem.Shared

function deserialize_results(file_path::String)
    results = nothing
    open(file_path, "r") do io
        results = deserialize(io)
    end
    return results
end

#display struct with the names of the properties and values 
function display_struct(str::Any)
    println
    type_of_struct = typeof(str)
    for name in fieldnames(type_of_struct)
        println(name, ": ", getfield(str, name))
    end
end

function print_file(path)
    println("path: ", path)
    results = deserialize_results(path)
    println("options: ") 
    display_struct(results["Options"])
    println("timings: ")
    display_struct(results["Timings"])
    println("energy:")
    println(results["Energy"])
    println("Converged")
    println(results["Converged"])
    println("---------------------------------")
end

for i in 1:22
    println("file = s22 $i")

    #convert i to XX format string with leading zeros
    i_str = lpad(i, 2, "0")

    # path = "/home/ac.jhayes/source/JuliaChem.jl/JuliaChem-Papers/DF-RHF-Benchmark/SingleNode_S22_RHF_vs_ScreenedDF/results/$(i_str)_MP2_DF_GUESS-iter-1.data"
    # print_file(path)

    path = "/home/ac.jhayes/source/JuliaChem.jl/JuliaChem-Papers/DF-RHF-Benchmark/SingleNode_S22_RHF_vs_ScreenedDF/results/$(i_str)_MP2_DF_RHF_SCREENED-iter-1.data"
    print_file(path)

    path = "/home/ac.jhayes/source/JuliaChem.jl/JuliaChem-Papers/DF-RHF-Benchmark/SingleNode_S22_RHF_vs_ScreenedDF/results/$(i_str)_MP2_RHF_DYNAMIC-iter-1.data"
    print_file(path)

end





# println("options: ")
# display_struct(results.Energy["Energy"])
# println("timings: ")
# display_struct(results["Timings"])
# println("energy:")
# println(results["Energy"])



