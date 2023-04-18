function check_d_tilde_array(array_1, array_2, index)
    any_found = false
    for i in 1:length(array_1)
        if array_1[i] != array_2[i]
            any_found = true
            println("found one that doesn't match in μ = $(index): $i $(array_1[i]) != $(array_2[i])")
        end
    end
    if !any_found
        println("no differences found in μ = $(index)")
    end
end

function read_array_in(path)
    io = open(path, "r")   
    d_tilde_array = []
    while !eof(io)
        numbers = Array{Float64, 1}()
        line = readline(io)
        line_parts = split(line, " = ")
        array_part = line_parts[3]
        array_part = replace(array_part, "[" => "")
        array_part = replace(array_part, "]" => "")
        array_part = replace(array_part, ";" => "")

        array_parts = split(array_part, " ")
        for str in array_parts
            number = parse(Float64, str)
            push!(numbers, number)
        end
        push!(d_tilde_array, numbers)
    end
    return d_tilde_array
end

function main(path1, path2)
    d_tilde1 = read_array_in(path1)
    d_tilde_2 = read_array_in(path2)
    for i in 1:length(d_tilde1)
        check_d_tilde_array(d_tilde1[i], d_tilde_2[i], i)
    end

end

main("/home/jackson/source/JuliaChem.jl/testoutputs/D_tilde_oneproc.log", "/home/jackson/source/JuliaChem.jl/testoutputs/D_tilde_twoproc.log")