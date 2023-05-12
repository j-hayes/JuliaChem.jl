mutable struct SCFData
    D 
    D_permuted
    D_tilde 
    D_tilde_permuted 
    two_electron_fock
    two_electron_fock_GPU
    coulomb_intermediate
    density
    occupied_orbital_coefficients
    μ :: Int
    occ :: Int
    A :: Int
end

function SCFData()
   
    return SCFData([], [], [], [], [], [], [], [],[] , 0, 0, 0)
end

export SCFData