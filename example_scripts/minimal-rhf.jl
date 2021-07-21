#=============================#
#== put needed modules here ==#
#=============================#
import JuliaChem

#================================#
#== JuliaChem execution script ==#
#================================#
include("minimal-rhf-repl.jl")

JuliaChem.initialize()
minimal_rhf("/home/jackson/Source/JuliaChem.jl/example_inputs/H2.json")
JuliaChem.finalize()

