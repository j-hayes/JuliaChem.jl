#=============================#
#== put needed modules here ==#
#=============================#
import JuliaChem
import Test
include("../example_scripts/full-rhf-repl.jl")


#==================================================================
 Script to check if the Density fitted method 
 values are close to the ones produced by non density fitted RHF
==================================================================#

function check_density_fitted_method_matches_RHF(denity_fitted_input_file, input_file)
  try 
    JuliaChem.initialize() 
    
    @time energy, properties = full_rhf(input_file)      
    @time density_fitted_energy, density_fitted_properties = full_rhf(denity_fitted_input_file)

    println("RHF energy   : $(energy["Energy"])")
    println("DF-RHF energy: $(density_fitted_energy["Energy"])")

    Test.@test energy["Energy"] ≈ density_fitted_energy["Energy"] atol=.00015 #15 micro hartree tolerance
    println("Test run successfully!")
  catch e
    println("check_density_fitted_method_matches_RHF Failed with exception:") 
    display(e) 
  end 
  JuliaChem.finalize()   
  
end
# check_density_fitted_method_matches_RHF(ARGS[1], ARGS[2])
# check_density_fitted_method_matches_RHF("./example_inputs/density_fitting/water_density_fitted.json", "./example_inputs/density_fitting/water_rhf.json")
# check_density_fitted_method_matches_RHF("./example_inputs/density_fitting/15_MP2_df.json", "./example_inputs/density_fitting/15_MP2.json")
# check_density_fitted_method_matches_RHF("./example_inputs/density_fitting/H2_density_fitted.json", "./example_inputs/density_fitting/H2_rhf.json")
check_density_fitted_method_matches_RHF("./example_inputs/density_fitting/22_MP2_df.json", 
"./example_inputs/density_fitting/22_MP2.json")
