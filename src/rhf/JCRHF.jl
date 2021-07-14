"""
  module JCRHF
The module required for computation of the wave function using the *Restricted
Hartree-Fock* (RHF) method in a Self-Consistent Field (SCF) calculation. This
module will be used often, as the RHF wave function is often the zeroth-order
wave function for closed-shell systems.
"""
module JCRHF

#== RHF Constants module ==#
Base.include(@__MODULE__,"./Constants.jl")
export Constants

#== RHF energy module ==#
Base.include(@__MODULE__,"energy/Energy.jl")
export Energy

#== RHF gradient module ==#
Base.include(@__MODULE__,"gradient/Gradient.jl")
export Gradient

#== RHF properties module ==#
Base.include(@__MODULE__,"properties/Properties.jl")
export Properties

end
