#=============================#
#== put needed modules here ==#
#=============================#
import JuliaChem


function run_julia_chem(molecule, driver, model, keywords)
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
function run_df_rhf(input_file, basis = "", auxilliary_basis = "", df_is_guess = false, guess = "", contraction_mode = "", load = "")
  try
    #== read in input file ==#
    molecule, driver, model, keywords = JuliaChem.JCInput.run(input_file;       
      output=0)       
    
    if length(basis) > 0
       model["basis"] = basis
    end
    
    if length(auxilliary_basis) > 0
      model["auxiliary_basis"] = auxilliary_basis
    end

    if length(load) > 0 
        keywords["scf"]["load"] = load #todo use constant
    end
    
    if df_is_guess
      keywords["scf"]["scf_type"] = "rhf" #todo use constant    
      keywords["scf"]["guess"] = "df" #todo use constant    
      keywords["scf"]["df_dele"] = 1E-3 #todo use constant
      keywords["scf"]["df_drms"] = 1E-3 #todo use constant
    else
      keywords["scf"]["scf_type"] = "df" #todo use constant
      keywords["scf"]["df_dele"] = 1E-6 #todo use constant
      keywords["scf"]["df_drms"] = 1E-6 #todo use constant   
      if length(guess) > 0
        keywords["scf"]["guess"] = guess #todo use constant
      end
    end

    if length(contraction_mode) > 0
      keywords["scf"]["contraction_mode"] = contraction_mode #todo use constant
    end

    rhf_energy, rhf_properties = run_julia_chem(molecule, driver, model, keywords)

    return (Energy = rhf_energy, Properties = rhf_properties, Keywords = keywords, Model = model, Molecule = molecule) 
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
function run_rhf(input_file, basis = "", guess = "", load="")
  try
    #== read in input file ==#
    molecule, driver, model, keywords = JuliaChem.JCInput.run(input_file;       
      output=0)   
      
    if length(basis) > 0
        model["basis"] = basis
    end 
    if length(guess) > 0
      keywords["scf"]["guess"] = guess #todo use constant
    end
    if length(load) > 0 
        keywords["scf"]["load"] = load #todo use constant

    end

    rhf_energy, rhf_properties = run_julia_chem(molecule, driver, model, keywords)

    return (Energy = rhf_energy, Properties = rhf_properties, Keywords = keywords, Model = model, Molecule = molecule) 
  catch e                                                                       
    bt = catch_backtrace()                                                      
    msg = sprint(showerror, e, bt)                                              
    println(msg)                                                                
                                                                                
    JuliaChem.finalize()                                                        
    exit()                                                                      
  end   
end


