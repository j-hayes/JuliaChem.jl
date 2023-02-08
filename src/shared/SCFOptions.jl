using JuliaChem.Shared.Constants
using MPI
mutable struct SCFOptions 
    density_fitting :: Bool 
    contraction_mode :: String
    load :: String 
end 

function create_default_scf_options()
    return SCFOptions(
        false, # density_fitting
        SCF_Keywords.ContractionMode.default,
        SCF_Keywords.Load.default
        )
end

function create_scf_options(density_fitting :: Bool, contraction_mode :: String, load::String)    
    return SCFOptions(density_fitting, contraction_mode, load)
end

function print_scf_options(options::SCFOptions, output ::Int)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0 && output >= 2
        println("SCF Options:")
        println("------------------------------")
        println("Density Fitting: ", options.density_fitting)
        println("Contraction Mode: ", options.contraction_mode)
        println("Integral Load: ", options.load)
        println("------------------------------") 
    end
end

export SCFOptions, create_default_scf_options, create_scf_options, print_scf_options