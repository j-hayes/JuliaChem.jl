#=============================#
#== put needed modules here ==#
#=============================#
import JuliaChem
using JuliaChem.JCModules

#================================#
#== JuliaChem execution script ==#
#================================#
function full_rhf(input_file)
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
  println("       For questions on usage, email David Poole at davpoole@iastate.edu.       ")
  println("                             Jackson                                            ")
  println("--------------------------------------------------------------------------------")
 
  try
    #== read in input file ==#
    molecule, driver, model, keywords = JuliaChem.JCInput.run(input_file;       
      output=2)       
    
    #== generate basis set ==#
    mol, basis = JuliaChem.JCBasis.run(molecule, model; 
      output=2) 
   
    #== molecule info ==#
    JuliaChem.JCMolecule.run(mol)

    #== calculation driver ==# 
    if driver == "energy"
        #== perform scf calculation ==#
        if haskey(keywords, "scf")
          method = "RHF"
          if !(method == "RHF" || method == "DFRHF")
            throw("Exception: Methods other than RHF and DFRFH are not supported yet!")
          end
          rhf_energy = JuliaChem.JCRHF.Energy.run(mol, basis, keywords["scf"]; 
            output=2,) 
        else
          rhf_energy = JuliaChem.JCRHF.Energy.run(mol, basis; 
            output=2) 
        end    
        #== compute molecular properties such as dipole moment ==#
        properties = JuliaChem.JCRHF.Properties.run(mol, basis, rhf_energy, 
          keywords["prop"]; output=2)
          return rhf_energy, properties
        
    else
      throw("Exception: Only energy calculations are currently supported!")
    end
  catch e                                                                       
    bt = catch_backtrace()                                                      
    msg = sprint(showerror, e, bt)                                              
    println(msg)          
    exit()                                                                                                                            
  end 

  println("--------------------------------------------------------------------------------")
  println("                      Your calculation has run to completion!                   ")
  println("                                                                                ")
  println("                       ========================================                 ")
  println("                                   HAVE A NICE DAY!                             ")
  println("                       ========================================                 ")
  println("--------------------------------------------------------------------------------")
end