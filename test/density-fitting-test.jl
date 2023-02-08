#=============================#
#== put needed modules here ==#
#=============================#
import JuliaChem
import Test
import MPI
include("../example_scripts/full-rhf-repl.jl")


#==================================================================
 Script to check if the Density fitted method 
 values are close to the ones produced by non density fitted RHF
==================================================================#

function check_density_fitted_method_matches_RHF(denity_fitted_input_file, input_file)
  try 
    JuliaChem.initialize() 
    #first runs not timed
    density_fitted_energy, density_fitted_properties = full_rhf(denity_fitted_input_file)
    energy, properties = full_rhf(input_file)      

    println("RUN 2 DF-RHF")
    DF_time = @elapsed begin @time begin 
      density_fitted_energy, density_fitted_properties = full_rhf(denity_fitted_input_file)
    end end
    MPI.Barrier(MPI.COMM_WORLD)
    RHF_time = @elapsed begin @time begin 
      energy, properties = full_rhf(input_file)      
    end end
    MPI.Barrier(MPI.COMM_WORLD)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
      println("RHF energy   : $(energy["Energy"]), time: $(RHF_time) seconds")
      println("DF-RHF energy: $(density_fitted_energy["Energy"]) time: $(DF_time) seconds")

      Test.@test energy["Energy"] ≈ density_fitted_energy["Energy"] atol=.00015 #15 micro hartree tolerance
      println("Test run successfully!")
    end
  catch e
    println("check_density_fitted_method_matches_RHF Failed with exception:") 
    display(e) 
  end 
  JuliaChem.finalize()   
  
end
# check_density_fitted_method_matches_RHF(ARGS[1], ARGS[2])
check_density_fitted_method_matches_RHF("./example_inputs/density_fitting/water_density_fitted.json", "./example_inputs/density_fitting/water_rhf.json")
# check_density_fitted_method_matches_RHF("./example_inputs/density_fitting/H2_density_fitted.json", "./example_inputs/density_fitting/H2_rhf.json")
# MP2_Num = 22
# check_density_fitted_method_matches_RHF("./example_inputs/density_fitting/$(MP2_Num)_MP2_df.json", "./example_inputs/density_fitting/$(MP2_Num)_MP2.json")
