#Base.include(@__MODULE__,"../basis/BasisStructs.jl")

"""
  module Properties
"""
module Properties 

include("Mulliken.jl")
include("Multipole.jl")
include("OrbitalEnergies.jl")

using JuliaChem.JCModules
using JuliaChem.JERI

using JSON
using MPI
using Printf

function run(mol::Molecule, basis::Basis, rhf_energy, 
  keywords; output="none")
  
  comm=MPI.COMM_WORLD

  if MPI.Comm_rank(comm) == 0 && output == "verbose"
      println("--------------------------------------------------------------------------------")
      println("                       ========================================                 ")
      println("                                RESTRICTED CLOSED-SHELL                         ")
      println("                                HARTREE-FOCK PROPERTIES                         ")
      println("                       ========================================                 ")
      println("")
  end
  
  #== create properties dict ==#
  properties = Dict{String, Any}([])
  
  #== compute dipole is selected ==#
  if haskey(keywords, "multipole")
    if keywords["multipole"] == "dipole"
      #== initial setup ==#
      jeri_prop_engine = JERI.PropEngine(mol.mol_cxx, 
        basis.basis_cxx) 

      P = rhf_energy["Density"] 

      #== compute dipole moment ==#
      if MPI.Comm_rank(comm) == 0 && output == "verbose"
        println("----------------------------------------          ")
        println("     Computing multiple moments...                ")
        println("----------------------------------------          ")
        println(" ")
        println("Dipole:       X           Y           Z         Tot. (D)        ") 
      end  
  
      dipole = compute_dipole(mol, basis, P, jeri_prop_engine)
      dipole_moment = sqrt(dipole[1]^2 + dipole[2]^2 + dipole[3]^2)
  
      @printf("          %.6f   %.6f    %.6f    %.6f     \n", 
        dipole[1], dipole[2], dipole[3], dipole_moment)
      println()
 
      properties["Dipole"] = (x = dipole[1], y = dipole[2], z = dipole[3], 
        moment = dipole_moment)  
    end
  end

  #== compute mulliken charges if selected ==#
  if haskey(keywords, "mulliken")
    if keywords["mulliken"] == true
      #== initial setup ==#
      D = rhf_energy["Density"] 
      S = rhf_energy["Overlap"]

      #== compute mulliken charges ==# 
      if MPI.Comm_rank(comm) == 0 && output == "verbose"
        println("----------------------------------------          ")
        println("     Computing mulliken charges...                ")
        println("----------------------------------------          ")
        println(" ")
        #println("Dipole:       X           Y           Z         Tot. (D)        ") 
      end  
  
      mulliken = compute_mulliken_charges(mol, basis, D, S)
      display(mulliken); println()
  
      #@printf("          %.6f   %.6f    %.6f    %.6f     \n", 
      #  dipole[1], dipole[2], dipole[3], dipole_moment)
  
      properties["Mulliken"] = mulliken
    end
  end
  
  if haskey(keywords, "mo energies")
    if keywords["mo energies"] == true
      #== initial setup ==#
      F = rhf_energy["Fock"]
      C = rhf_energy["MO Coeff"]

      #== convert Fock matrix to MO basis ==#
      F_mo = compute_orbital_energies(F, C)

      #== print orbital energies ==#
      if MPI.Comm_rank(comm) == 0 && output == "verbose" 
        println("----------------------------------------          ")           
        println(" Computing molecular orbital energies...           ")           
        println("----------------------------------------          ")           
        println(" ")                                                            
        println("Orbital #     Orbital energy")                                 
        for index::Int64 in 1:size(F_mo)[1] 
            println("   ",index,"       ",F_mo[index,index])                    
        end                                                                     
        println(" ")                                                            
      end

      #== compute HOMO-LUMO gap ==#
      nocc = basis.nels >> 1 
      
      homo_pos = nocc
      E_homo = F_mo[homo_pos, homo_pos]
      
      lumo_pos = homo_pos + 1
      E_lumo = F_mo[lumo_pos, lumo_pos]
  
      homo_lumo_gap = abs(E_homo - E_lumo)
 
      if MPI.Comm_rank(comm) == 0 && output == "verbose" 
        println("----------------------------------------          ")           
        println("        Computing HOMO-LUMO gap...                ")
        println("----------------------------------------          ")           
        println(" ")                                                            
        println("The HOMO is located at MO orbital #$homo_pos,")
        println("  with an energy of $E_homo h.")
        println(" ")                                                            
        println("The LUMO is located at MO orbital #$lumo_pos,")
        println("  with an energy of $E_lumo h.")
        println(" ")                                                            
        println("HOMO-LUMO gap: $homo_lumo_gap h")
        println(" ")                                                            
      end

      properties["MO Energies"] = (energies = F_mo, homo_lumo = homo_lumo_gap)
    end    
  end

  if MPI.Comm_rank(comm) == 0 && output == "verbose"
    println("                       ========================================                 ")
    println("                              END RESTRICTED CLOSED-SHELL                       ")
    println("                                HARTREE-FOCK PROPERTIES                         ")
    println("                       ========================================                 ")
    println("--------------------------------------------------------------------------------")
  end

  return properties
end
export run

end
