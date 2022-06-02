"""
  module JCRHF
The module required for computation of the wave function using the *Restricted
Hartree-Fock* (RHF) method in a Self-Consistent Field (SCF) calculation. This
module will be used often, as the RHF wave function is often the zeroth-order
wave function for closed-shell systems.
"""
module Energy 

using JuliaChem.JCModules
using JuliaChem.JCRHF.Constants
using JuliaChem.JERI

using MPI
using JSON

Base.include(@__MODULE__,"EnergyHelpers.jl")
Base.include(@__MODULE__,"SCF.jl")

"""
  overload to allow old mehtods that don't use auxillary basis sets to not need to be changed
"""
function run(mol::Molecule, basis::Basis,
  scf_flags = Dict(); output=0)
  basis_sets = CalculationBasisSets(basis, nothing)
  return run(mol, basis_sets, scf_flags; output=output)
end 


"""
  run(input_info::Dict{String,Dict{String,Any}}, basis::Basis)

Execute the JuliaChem RHF algorithm.

One input variable is required:
1. input_info = Information gathered from the input file.
2. basis_sets = The wrapper for basis set structs which contain basis shells, determined from the input file.

One variable is output:
1. scf = Data saved from the SCF calculation.

Thus, proper use of the RHF.run() function would look like this:

```
scf = RHF.run(input_info, basis)
```
"""
function run(mol::Molecule, basis_sets::CalculationBasisSets, scf_flags = Dict(); output=0)
  
  comm=MPI.COMM_WORLD

  if MPI.Comm_rank(comm) == 0 && output >= 2
    println("--------------------------------------------------------------------------------")
    println("                       ========================================                 ")
    println("                                RESTRICTED CLOSED-SHELL                         ")
    println("                                  HARTREE-FOCK ENERGY                           ")
    println("                       ========================================                 ")
    println("")
  end

  #== actually perform scf calculation ==#
  rhfenergy = rhf_energy(mol, basis_sets, scf_flags; output=output)

  if MPI.Comm_rank(comm) == 0 && output >= 2
    println("                       ========================================                 ")
    println("                              END RESTRICTED CLOSED-SHELL                       ")
    println("                                  HARTREE-FOCK ENERGY                           ")
    println("                       ========================================                 ")
    println("--------------------------------------------------------------------------------")
  end

  return rhfenergy 
end
export run

end
