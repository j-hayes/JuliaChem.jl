module Constants

module SCF_Keywords
    module SCFType
        const scf_type = "scf_type"
        const rhf = "rhf" # Restricted Hartree Fock
        const density_fitting = "df" # Density Fitted Restricted Hartree Fock
    end

    module Guess
        const guess = "guess"
        const default = "hcore"
        const hcore = "hcore"
        const density_fitting = "df" # density fitting with hcore guess before using full rhf 
        const sad = "sad"
    end

    module Convergence
        const density_fitting_energy = "df_dele"
        const energy = "dele"
        const energy_default = 1E-3

        const density_fitting_density = "df_rmsd"
        const density = "rmsd"
        const density_default = 1E-3

        const max_iterations = "niter"
        const max_iterations_default = 10

        const df_max_iterations = "df_niter"
        const df_max_iterations_default = 10
    end

    module ContractionMode 
        const contraction_mode = "contraction_mode"
        const default = "TensorOperations"
        const tensor_operations = "TensorOperations" # use TensorOperations.jl library 
        const BLAS = "BLAS" # use BLAS library 
    end 

    module IntegralLoad 
        const load = "load"
        const default = "static"
        const sequential = "sequential"
        const dynamic = "dynamic"
    end
    
    export SCFType, ContractionMode, IntegralLoad, Guess, Convergence
end 



export SCF, SCF_Keywords

end