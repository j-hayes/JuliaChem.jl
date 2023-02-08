#=============================#
#== put needed modules here ==#
#=============================#
import JuliaChem
import Test
import MPI
using JuliaChem.Shared
include("../example_scripts/full-rhf-repl.jl")


#==================================================================
 Script to check if the Density fitted method 
 values are close to the ones produced by non density fitted RHF
==================================================================#

function check_density_fitted_method_matches_RHF(denity_fitted_input_file, input_file)
  try 
    JuliaChem.initialize() 
    #first runs not timed
    df_scf_results, density_fitted_properties = full_rhf(denity_fitted_input_file)
    scf_results, properties = full_rhf(input_file)      

    println("RUN 2 DF-RHF")
    DF_time = @elapsed begin @time begin 
      df_scf_results, density_fitted_properties = full_rhf(denity_fitted_input_file)
    end end
    MPI.Barrier(MPI.COMM_WORLD)
    RHF_time = @elapsed begin @time begin 
      scf_results, properties = full_rhf(input_file)      
    end end
    MPI.Barrier(MPI.COMM_WORLD)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0

      println("---------------------------------------------")
      println("RHF iteration Times (seconds):")
      print_iteration_times(scf_results["Timings"])
      
      println("DF-RHF iteration Times (seconds):")
      print_iteration_times(df_scf_results["Timings"])

    
      println("---------------------------------------------")

      println("RHF energy   : $(scf_results["Energy"]), time: $(RHF_time) seconds")
      println("DF-RHF energy: $(df_scf_results["Energy"]) time: $(DF_time) seconds")

      Test.@test scf_results["Energy"] ≈ df_scf_results["Energy"] atol=.00015 #15 micro hartree tolerance
      println("Test run successfully!")
    end
  catch e
    println("check_density_fitted_method_matches_RHF Failed with exception:\n") 
    display(e) 
  end 
  JuliaChem.finalize()   
  
end


# check_density_fitted_method_matches_RHF(ARGS[1], ARGS[2])
# check_density_fitted_method_matches_RHF("./example_inputs/density_fitting/water_density_fitted.json", "./example_inputs/density_fitting/water_rhf.json")
# check_density_fitted_method_matches_RHF("./example_inputs/density_fitting/H2_density_fitted.json", "./example_inputs/density_fitting/H2_rhf.json")
MP2_Num = "01"
check_density_fitted_method_matches_RHF("./example_inputs/density_fitting/$(MP2_Num)_MP2_df.json", "./example_inputs/density_fitting/$(MP2_Num)_MP2.json")
