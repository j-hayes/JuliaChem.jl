module JuliChem

Base.include(@__MODULE__,"src/input/input_interface.jl")
Base.include(@__MODULE__,"src/molecule/molecule_interface.jl")
Base.include(@__MODULE__,"src/rhf/rhf_interface.jl")

#------------------------------#
#           Script.jl          #
#------------------------------#
#Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
function julia_main()
    println("                       ========================================                ")
    println("                                 Welcome to JuliChem!                          ")
    println("                        JuliChem is a software package written                 ")
    println("                          in Julia for the purpose of quantum                  ")
    println("                                 chemical calculations.                        ")
    println("                             Let's get this party started!                     ")
    println("                       ========================================                ")
    println(" ")
    println("                                 Authors: David Poole                          ")
    println(" ")
    #read in input file
    @time flags::Flags, coord::Array{Float64,2} = do_input()

    #analyze molecular coordinates
    @time do_coordinate_analysis(coord)

    #perform scf calculation
    #@time scf::Data = do_rhf(flags)

    #we have run to completion! :)
    println("--------------------------------------------------------------------------------------")
    println("                       ========================================                       ")
    println("                        The calculation has run to completion!                        ")
    println("                                       Sayonara!                                      ")
    println("                       ========================================                       ")
#    return 0
end

#we want to precompile all involved modules to reduce cold runs
#include("snoop/precompile_Base.jl")
#_precompile_()
#include("snoop/precompile_Core.jl")
#_precompile_()
#include("snoop/precompile_Distributed.jl")
#_precompile_()
#include("snoop/precompile_LinearAlgebra.jl")
#_precompile_()
#include("snoop/precompile_JuliChem.jl")
#_precompile_()
#include("snoop/precompile_unknown.jl")
#_precompile_()

end
