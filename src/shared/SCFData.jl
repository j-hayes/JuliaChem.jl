mutable struct SCFData
    D :: AbstractArray
    D_tilde :: AbstractArray
    D_tilde_permuted :: AbstractArray
    two_electron_fock :: AbstractArray
    μ :: Int
    occ :: Int
    A :: Int
end

export SCFData