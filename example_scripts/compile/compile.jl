using PackageCompiler
try
    println("Threads.nthreads = ",Threads.nthreads())
catch 
    println("Threads not available")
end

path_to_so=ENV["JULIACHEM_SYSIMG_PATH"]
path_to_precompile_script=ENV["JULIACHEM_PRECOMPILE_SCRIPT_PATH"]

println("Creating sysimage at $(path_to_so) using precompile script at $(path_to_precompile_script)")
flush(stdout)
PackageCompiler.create_sysimage(; sysimage_path=path_to_so,
                                                     precompile_execution_file=path_to_precompile_script)
println("done creating sysimage");flush(stdout)

