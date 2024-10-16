#=============================#
#== put needed modules here ==#
#=============================#
import JuliaChem
import MPI
using JuliaChem.JCModules
using JuliaChem.Shared.JCTC

#================================#
#== JuliaChem execution script ==#
#================================#
function full_rhf(input_file; output=3)

  if MPI.Comm_rank(MPI.COMM_WORLD) == 0

  println("--------------------------------------------------------------------------------")
  println("                       ========================================                 ")
  println("                               WELCOME TO JULIACHEM.JL!                         ")
  println("                       ========================================                 ")
  println("                                                                                ")
  println("              JuliaChem.jl is an electronic structure theory package            ")
  println("         developed by the Gordon group at Iowa State University. It is          ")
  println("        designed to apply the strengths of the Julia programming language       ")
  println("          (high-level semantics and low-level performance) to a quantum         ")
  println("                                chemistry package.                              ")
  println("                                                                                ")
  println("         For more information, refer to the following paper: Poole, D.          ")
  println("        Galvaz Vallejo, J. L.; Gordon, M. S. \"A New Kid on the Block:          ")
  println("         Application of Julia to Hartree-Fock Calculations.\" J. Chem.          ") 
  println("                     Theory Compute. 2020, 16, 8, 5006-5013.                    ")
  println("                                                                                ")
  println("                                                                                ")
  println("           Density Fitting Restricted Hartree Fock                              ")
  println("               implemented by Jackson Hayes <jhayes1@iastate.edu                ")
  println("                                                                                ")
  println("       For questions on usage, email David Poole at davpoole@iastate.edu.       ")
  println("                                                                                ")
  println("--------------------------------------------------------------------------------")
  end 
  rhf_energy = nothing
  properties = nothing
  try
    #== read in input file ==#
    molecule, driver, model, keywords = JuliaChem.JCInput.run(input_file;       
      output=output)       
    
    #== generate basis set ==#
    mol, basis = JuliaChem.JCBasis.run(molecule, model; 
      output=output) 
   
    #== molecule info ==#
    JuliaChem.JCMolecule.run(mol)

    #== calculation driver ==# 
    
    if driver == "energy"
        #== perform scf calculation ==#
        if haskey(keywords, "scf")         
          rhf_energy = JuliaChem.JCRHF.Energy.run(mol, basis, keywords["scf"]; 
            output=output) 
        else
          rhf_energy = JuliaChem.JCRHF.Energy.run(mol, basis; 
            output=output) 
        end    
        #== compute molecular properties such as dipole moment ==#
        properties = JuliaChem.JCRHF.Properties.run(mol, basis, rhf_energy, 
          keywords["prop"]; output=output)
        
    else
      throw("Exception: Only energy calculations are currently supported!")
    end
  catch e                                                                       
    bt = catch_backtrace()                                                      
    msg = sprint(showerror, e, bt)                                              
    println(msg)          
    exit()                                                                                                                            
  end 
  
  if MPI.Comm_rank(MPI.COMM_WORLD) == 0
  println("--------------------------------------------------------------------------------")
  println("                      Your calculation has run to completion!                   ")
  println("                                                                                ")
  println("                       ========================================                 ")
  println("                                   HAVE A NICE DAY!                             ")
  println("                       ========================================                 ")
  println("--------------------------------------------------------------------------------")
  end
  return rhf_energy, properties
end