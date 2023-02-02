module JCModules

using StaticArrays
using CxxWrap
using JuliaChem.JERI

Base.include(@__MODULE__,"Constants.jl")

Base.include(@__MODULE__,"Globals.jl")

Base.include(@__MODULE__,"BasisStructs.jl")

Base.include(@__MODULE__,"SCFOptions.jl")

Base.include(@__MODULE__,"MolStructs.jl")

end 
