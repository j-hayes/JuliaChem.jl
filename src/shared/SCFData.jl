mutable struct SCFData
    D 
    D_permuted
    D_tilde 
    D_tilde_permuted 
    two_electron_fock
    coulomb_intermediate
    μ :: Int
    occ :: Int
    A :: Int
end

export SCFData