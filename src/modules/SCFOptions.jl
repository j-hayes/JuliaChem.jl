using JuliaChem.JCModules.Constants

struct SCFOptions
    load :: String
    density_fitting :: Bool 
    contraction_library :: String
end

function build_SCFOptions(load = SCF.default_load, density_fitting = false, contraction_library = SCF.default_contract)
    return SCFOptions(load, density_fitting, contraction_library)
end

export SCFOptions, build_SCFOptions