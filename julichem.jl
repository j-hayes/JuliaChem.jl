module JuliChem

Base.include(@__MODULE__,"src/input/input_interface.jl")
Base.include(@__MODULE__,"src/rhf/rhf_interface.jl")

#------------------------------#
#           Script.jl          #
#------------------------------#
Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    println("========================================")
    println("Welcome to JuliChem!")
    println("JuliChem is a software package written")
    println("in Julia for the purpose of quantum")
    println("chemical calculations.")
    println("Let's get this party started!")
    println("========================================")
    println(" ")

    #read in input file
    @time dat::Array{String,1}, flags::Flags = do_input(ARGS[1])

    #perform scf calculation
    @time scf::Data = do_rhf(dat,flags)

    #we have run to completion! :)
    println("========================================")
    println("The calculation has run to completion!")
    println("Sayonara!")
    println("========================================")
    return 0
end

#we want to precompile all involved modules to reduce cold runs
include("snoop/precompile_Base.jl")
_precompile_()
#include("snoop/precompile_Core.jl")
#_precompile_()
include("snoop/precompile_LinearAlgebra.jl")
_precompile_()
include("snoop/precompile_JuliChem.jl")
_precompile_()
include("snoop/precompile_unknown.jl")
_precompile_()

end
