#=============================#
#== put needed modules here ==#
#=============================#
import JuliaChem


function run_travis_rhf(molecule, driver, model, keywords)
  #== generate basis set ==#
  mol, basis = JuliaChem.JCBasis.run(molecule, model; output=2)          

  #== compute molecular inform ation ==#
  JuliaChem.JCMolecule.run(mol)

  #== perform scf calculation ==#
  rhf_energy = JuliaChem.JCRHF.Energy.run(mol, basis, keywords["scf"]; 
    output=0) 

  #== compute molecular properties ==# 
  rhf_properties = JuliaChem.JCRHF.Properties.run(mol, basis, rhf_energy, keywords["prop"],
    output=0)  

  return rhf_energy, rhf_properties
end


#================================#
#== JuliaChem execution script ==#
#================================#
function travis_rhf_density_fitting(input_file, auxilliary_basis)
  try
    #== read in input file ==#
    molecule, driver, model, keywords = JuliaChem.JCInput.run(input_file;       
      output=2)       
    
    model["auxiliary_basis"] = auxilliary_basis
    keywords["scf"]["scf_type"] = "df" #todo use constant    
    rhf_energy, rhf_properties = run_travis_rhf(molecule, driver, model, keywords)

    return (Energy = rhf_energy, Properties = rhf_properties) 
  catch e                                                                       
    bt = catch_backtrace()                                                      
    msg = sprint(showerror, e, bt)                                              
    println(msg)                                                                
                                                                                
    JuliaChem.finalize()                                                        
    exit()                                                                      
  end   
end

#================================#
#== JuliaChem execution script ==#
#================================#
function travis_rhf(input_file)
  try
    #== read in input file ==#
    molecule, driver, model, keywords = JuliaChem.JCInput.run(input_file;       
      output=2)       
    
    rhf_energy, rhf_properties = run_travis_rhf(molecule, driver, model, keywords)

    return (Energy = rhf_energy, Properties = rhf_properties) 
  catch e                                                                       
    bt = catch_backtrace()                                                      
    msg = sprint(showerror, e, bt)                                              
    println(msg)                                                                
                                                                                
    JuliaChem.finalize()                                                        
    exit()                                                                      
  end   
end


