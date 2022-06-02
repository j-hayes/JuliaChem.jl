module Constants
    
module SCF
    const RHF = "rhf" # Restricted Hartree Fock
    const DensityFitting = "df" # Density Fitted Restricted Hartree Fock
end 

module SCF_Keywords
    SCF_TYPE = "scf_type"
end 

export SCF, SCF_Keywords

end