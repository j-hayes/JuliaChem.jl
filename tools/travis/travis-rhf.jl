#=============================#
#== put needed modules here ==#
#=============================#
import JuliaChem


function run_travis_rhf(molecule, driver, model, keywords)
  #== generate basis set ==#
  mol, basis = JuliaChem.JCBasis.run(molecule, model; output=0)          

  #== compute molecular inform ation ==#
  JuliaChem.JCMolecule.run(mol)


  #== perform scf calculation ==#
  @time begin
    start_time = time()
    rhf_energy = JuliaChem.JCRHF.Energy.run(mol, basis, keywords["scf"]; 
      output=0) 
    end_time = time()
    rhf_energy["Timings"].run_time = end_time - start_time
  end 
  #== compute molecular properties ==# 
  rhf_properties = JuliaChem.JCRHF.Properties.run(mol, basis, rhf_energy, keywords["prop"],
    output=0)  
  
  return rhf_energy, rhf_properties
end


#================================#
#== JuliaChem execution script ==#
#================================#
function travis_rhf_density_fitting(input_file, auxilliary_basis, df_is_guess = false)
  try
    #== read in input file ==#
    molecule, driver, model, keywords = JuliaChem.JCInput.run(input_file;       
      output=0)       
    
    model["auxiliary_basis"] = auxilliary_basis
    
    if df_is_guess
      keywords["scf"]["guess"] = "df" #todo use constant    
    else
      keywords["scf"]["scf_type"] = "df" #todo use constant
    end
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
      output=0)       
    
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


