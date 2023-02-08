module Constants

module SCF_Keywords
    module SCFType
        const scf_type = "scf_type"
        const rhf = "rhf" # Restricted Hartree Fock
        const density_fitting = "df" # Density Fitted Restricted Hartree Fock
    end

    module ContractionMode 
        const contraction_mode = "contraction_mode"
        const default = "default"
        const tensor_operations = "TensorOperations" # use TensorOperations.jl library 
    end 

    module IntegralLoad 
        const load = "load"
        const default = "static"
        const sequential = "sequential"
        const dynamic = "dynamic"
    end
    
    export SCFType, ContractionMode, IntegralLoad
end 



export SCF, SCF_Keywords

end