function main(file_path::String)
    number_of_inputs = 18-10+1
    polyglycine_index = 0
    total_gpu_seconds = Array{Array{Float64}}(undef, number_of_inputs)
    k_seconds = Array{Array{Float64}}(undef, number_of_inputs)
    number_of_basis_functions = Array{Int64}(undef, number_of_inputs)

    for i in 1:number_of_inputs
        total_gpu_seconds[i] = Array{Float64}(undef, 0)
        k_seconds[i] = Array{Float64}(undef, 0)
    end
    number_of_basis_functions .= 0
    open(file_path) do file
        for line in eachline(file)
            if occursin("Running polyglycine", line)
                polyglycine_index += 1
                total_gpu_seconds[polyglycine_index] = Array{Float64}(undef, 0)
            end
            if occursin("Number of basis functions:", line)
                line_splits = split(line)
                number_of_basis_functions[polyglycine_index] = parse(Int64, line_splits[end])

            end
            if occursin("total time:", line)
                line_splits = split(line)
                seconds = parse(Float64, line_splits[end])
                push!(total_gpu_seconds[polyglycine_index], seconds)
            end
            # get k time
            if occursin("K_time:", line)
                line_splits = split(line, "K_time:")
                with_kblcok = line_splits[2]
                line_splits = split(with_kblcok, " ")
                line = strip(line_splits[2])
                seconds = parse(Float64, line)
                push!(k_seconds[polyglycine_index], seconds)

            end
        end
    end
    # get the averages of the iteration times skippingt he firsti teration 
    println("polyglycine iterations $polyglycine_index")
    display(total_gpu_seconds)

    total_averages = Array{Float64}(undef, number_of_inputs)
    k_averages = Array{Float64}(undef, number_of_inputs)
    mins = Array{Float64}(undef, number_of_inputs)
    maxs = Array{Float64}(undef, number_of_inputs)

    for i in 1:number_of_inputs
        mins[i] = minimum(total_gpu_seconds[i])
        maxs[i] = maximum(total_gpu_seconds[i])
    end
    for i in 1:number_of_inputs
        # iteration_seconds[i] = remove_outlyers(iteration_seconds[i])
        if length(total_gpu_seconds[i]) > 1
            total_gpu_seconds[i] = total_gpu_seconds[i][2:end]
            k_seconds[i] = k_seconds[i][2:end]
        end
        total_averages[i] = sum(total_gpu_seconds[i]) / length(total_gpu_seconds[i])
        k_averages[i] = sum(k_seconds[i]) / length(k_seconds[i])
    end
    println("number of basis functions ")
    display(number_of_basis_functions)
   
    println("min  fock build time")
    display(mins)
    println("max fock build time")
    display(maxs)
    println("average fock build time")
    display(total_averages)
    println("average k time")
    display(k_averages)
end

function remove_outlyers(array::Array{Float64})
    mean = sum(array) / length(array)
    std = sqrt(sum((array .- mean).^2) / length(array))
    new_array = Array{Float64}(undef, 0)
    for i in 1:length(array)
        if abs(array[i] - mean) < 2 * std
            push!(new_array, array[i])
        end
    end
    return new_array

end


main(ARGS[1])

# using HDF5 

# function main()
#     # read in the J data 
#     # file_J_path_screened = "/pscratch/sd/j/jhayes1/source/JuliaChem.jl/testoutputs/coulomb_inter_1screened.h5"
#     # file_J_path_GPU = "/pscratch/sd/j/jhayes1/source/JuliaChem.jl/testoutputs/coulomb_gpu.h5"

#     # J_screened = h5read(file_J_path_screened, "coulomb")
#     # J_GPU = h5read(file_J_path_GPU, "coulomb")

#     # diff = J_screened - J_GPU

#     # println("max diff ", maximum(diff))
#     # display(diff)

#     file_exchange_path_screened = "/pscratch/sd/j/jhayes1/source/JuliaChem.jl/testoutputs/exchange_inter_1screened.h5"
#     file_exchange_path_GPU = "/pscratch/sd/j/jhayes1/source/JuliaChem.jl/testoutputs/exchange_gpu.h5"

#     exchange_screened = h5read(file_exchange_path_screened, "exchange")
#     exchange_GPU = h5read(file_exchange_path_GPU, "exchange")

#     diff = exchange_screened - exchange_GPU
    
#     println("max diff ", maximum(diff))
#     display(diff)



# end

# main()

# using HDF5

# function main()
#     path_gpu = "/pscratch/sd/j/jhayes1/source/JuliaChem.jl/debug_fock_gpu_1.h5"
#     path_screened = "/pscratch/sd/j/jhayes1/source/JuliaChem.jl/debug_fock_cpu_J-1.h5"

#     fock_gpu = h5read(path_gpu, "fock")
#     fock_screened = h5read(path_screened, "fock")

#     diff = fock_gpu - fock_screened
#     display(diff)

#     max_diff = maximum(abs.(diff))
#     println("max diff ", max_diff)
# end

# main()