
function main(file_path::String, number_of_inputs_str::String, number_of_trials::Int)
    total_trial_count = parse(Int64, number_of_inputs_str)

    total_trial_count = total_trial_count*number_of_trials

    polyglycine_trial_index = 0
    polyglycine_index = 0
    total_gpu_seconds = Array{Array{Float64}}(undef, total_trial_count)
    k_seconds = Array{Array{Float64}}(undef, total_trial_count)
    number_of_basis_functions = Array{Int64}(undef, total_trial_count)

    for i in 1:total_trial_count
        total_gpu_seconds[i] = Array{Float64}(undef, 0)
        k_seconds[i] = Array{Float64}(undef, 0)
    end
    skip_first_iter = false
    number_of_basis_functions .= 0
    open(file_path) do file
        for line in eachline(file)

            if occursin("starting warm up", line) #skip warmups
                println("skipping warmup")
                line = readline(file)
                while !occursin("finished warm up", line)
                    line = readline(file)
                end
            end
            if occursin("Iter        Energy    ", line)
                skip_first_iter = true   
            end
            line_is_running_polyglycine = occursin("Running polyglycine", line) 
            if line_is_running_polyglycine 
                polyglycine_index += 1
            end
            if occursin("run trial:", line)
                if polyglycine_trial_index == total_trial_count
                    break
                end
                polyglycine_trial_index += 1    
            end
       
            if occursin("Number of basis functions:", line)
                line_splits = split(line)
                number_of_basis_functions[polyglycine_index] = parse(Int64, line_splits[end])

            end
            if occursin("gpu fock time ", line)
                line_splits = split(line)
                seconds = parse(Float64, line_splits[end])
                if skip_first_iter
                    skip_first_iter = false
                    continue
                end 
                println("pushing to trial $polyglycine_trial_index")
                push!(total_gpu_seconds[polyglycine_trial_index], seconds)
            end
            # get k time
            if occursin("K_time:", line)
                line_splits = split(line, "K_time:")
                with_kblcok = line_splits[2]
                line_splits = split(with_kblcok, " ")
                line = strip(line_splits[2])
                seconds = parse(Float64, line)
                push!(k_seconds[polyglycine_trial_index], seconds)
            end
        end
    end
    # get the averages of the iteration times skippingt he firsti teration 


    total_averages = Array{Float64}(undef, total_trial_count)
    k_averages = Array{Float64}(undef, total_trial_count)
    mins = Array{Float64}(undef, total_trial_count)
    maxs = Array{Float64}(undef, total_trial_count)

    for i in 1:total_trial_count
        println("trial $i")
        println("total gpu seconds ", total_gpu_seconds[i])
        mins[i] = minimum(total_gpu_seconds[i])
        maxs[i] = maximum(total_gpu_seconds[i])
    end
    for i in 1:total_trial_count
        # total_gpu_seconds[i] = remove_outlyers(total_gpu_seconds[i])
        # if length(total_gpu_seconds[i]) > 1
        #     total_gpu_seconds[i] = total_gpu_seconds[i][2:end]
        #     k_seconds[i] = k_seconds[i][2:end]
        # end
        total_averages[i] = sum(total_gpu_seconds[i]) / length(total_gpu_seconds[i])
        k_averages[i] = sum(k_seconds[i]) / length(k_seconds[i])
    end
    println("polyglycine iterations $polyglycine_trial_index")
    display(total_gpu_seconds)

    println("number of basis functions ")
    display(number_of_basis_functions)
   
    println("min  fock build time")
    display(mins)
    println("max fock build time")
    display(maxs)
    println("average k time")
    display(k_averages)

    println("average fock build time")
    display(total_averages)
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


number_of_trials = 1
if length(ARGS) >= 3
    number_of_trials = parse(Int64, ARGS[3])
end

main(ARGS[1], ARGS[2],  number_of_trials)

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