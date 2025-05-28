#get the number of processors and sockets from command lscpu 
function main()

    #run the ls command to get the number of processors and sockets
    cmd = `lscpu`
    output = read(`$cmd`, String)
    #parse the output 
    lines = split(output, "\n")

    ncpus = 0
    nsockets = 0
    ncores_per_soccet = 0
    nthreads_per_core = 0

    for line in lines
        line = strip(line)
        if startswith(line, "CPU(s):") && ncpus == 0
            ncpus = parse(Int, split(line, ":")[2])
        end
        if startswith(line, "Socket(s):") && nsockets == 0
            nsockets = parse(Int, split(line, ":")[2])
        end
        if startswith(line, "Core(s) per socket:") && ncores_per_soccet == 0
            ncores_per_soccet = parse(Int, split(line, ":")[2])
        end
        if startswith(line, "Thread(s) per core:") && nthreads_per_core == 0
            nthreads_per_core = parse(Int, split(line, ":")[2])
        end
        
    end

    total_cores=ncores_per_soccet*nsockets
    println("$ncpus, $nsockets, $ncores_per_soccet, $nthreads_per_core, $total_cores")

end

main()