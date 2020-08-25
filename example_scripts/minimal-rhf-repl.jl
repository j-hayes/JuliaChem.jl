#=============================#
#== put needed modules here ==#
#=============================#
import JuliaChem

#================================#
#== JuliaChem execution script ==#
#================================#
function minimal_rhf(input_file)
  try
    #== read in input file ==#
    molecule, driver, model, keywords = JuliaChem.JCInput.run(input_file;       
      output="none")       
    
    #== generate basis set ==#
    mol, basis = JuliaChem.JCBasis.run(molecule, model; 
      output="verbose")          

    JuliaChem.JCMolecule.run(mol)

    #== perform scf calculation ==#
    rhf_energy = JuliaChem.JCRHF.run(mol, basis, keywords["scf"]; 
      output="verbose") 
 
    #== perform gradient ==#
    rhf_gradient = JuliaChem.JCGrad.run(mol, basis; output="verbose")

    #== reset JuliaChem runtime ==#
    JuliaChem.reset()
    return rhf_energy, rhf_gradient
  catch e                                                                       
    bt = catch_backtrace()                                                      
    msg = sprint(showerror, e, bt)                                              
    println(msg)                                                                
                                                                                
    JuliaChem.finalize()                                                        
    exit()                                                                      
  end   
end
