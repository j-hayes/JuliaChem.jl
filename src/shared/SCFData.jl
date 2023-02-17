mutable struct SCFData
    xyK :: AbstractArray
    xiK :: AbstractArray
    two_electron_fock :: AbstractArray
    μ :: Int
    occ :: Int
    A :: Int
end

export SCFData