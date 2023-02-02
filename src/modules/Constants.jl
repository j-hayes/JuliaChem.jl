
module Constants
    module SCF
        const SCF_TYPE = "scf_type"
        const RHF = "rhf" # Restricted Hartree Fock
        const DensityFitting = "df" # Density Fitted Restricted Hartree Fock
    end 

    module Load
        const LOAD = "load"
        # load types 
        const default_load = "static"
        const static = "static"
        const sequential = "sequential"
    end
    module Contraction
        const TENSOR_CONTRACT = "contraction_library"
        # tensor contraction options 
        const Tensor_Op = "TensorOperations"
        const default_contract = "default"
    end 

    export SCF, Load, Contraction
end 
export Constants