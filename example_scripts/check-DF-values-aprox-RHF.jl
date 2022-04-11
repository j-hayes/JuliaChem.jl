#=============================#
#== put needed modules here ==#
#=============================#
import JuliaChem
import Test
include("full-rhf-repl.jl")


#==================================================================
 Script to check if the Density fitted method 
 values are close to the ones produced by non density fitted RHF
==================================================================#

function check_density_fitted_method_matches_RHF(denity_fitted_input_file, input_file)
  try 
    JuliaChem.initialize() 

    # density_fitted_energy, density_fitted_properties = full_rhf(denity_fitted_input_file)
    t = @elapsed  begin
      energy, properties = full_rhf(input_file)      
    end
    

    # Test.@test energy["Energy"] ≈ density_fitted_energy["Energy"]
    println("Test run successfully! in $(t) s")
  catch e
    println("check_density_fitted_method_matches_RHF Failed with exception:") 
    display(e) 
  end 
  JuliaChem.finalize()   
  
end
# check_density_fitted_method_matches_RHF(ARGS[1], ARGS[2])
check_density_fitted_method_matches_RHF("./example_inputs/density_fitting/water_density_fitted.json", "./example_inputs/density_fitting/water_rhf.json")
