module RHFStructs

#=
"""
    Data
Summary
======
Core Hartree-Fock data structures

Fields
======
1. Fock::Array{Float64,2} = Fock Matrix
2. Density::Array{Float64,2} = Density Matrix
3. Coeff::Array{Float64,2} = Molecular Orbital Coefficient Matrix
4. Energy::Float64 = Electronic Energy
"""
=#
mutable struct Data{T<:Number}
    Fock::Array{T,2}
    Density::Array{T,2}
    Coeff::Array{T,2}
    Energy::T
end
export Data

end
