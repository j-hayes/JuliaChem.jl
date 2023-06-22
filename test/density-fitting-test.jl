#=============================#
#== put needed modules here ==#
#=============================#
import JuliaChem
import Test
using JuliaChem.Shared
using MPI
include("../example_scripts/full-rhf-repl.jl")


#==================================================================
 Script to check if the Density fitted method 
 values are close to the ones produced by non density fitted RHF
==================================================================#

function check_density_fitted_method_matches_RHF(denity_fitted_input_file, input_file)
  try 
    JuliaChem.initialize() 

    #startup compilation runs
    df_scf_results, density_fitted_properties = full_rhf(joinpath(@__DIR__, "../example_inputs/density_fitting/water_density_fitted.json"))
    # scf_results, properties = full_rhf(joinpath(@__DIR__, "../example_inputs/density_fitting/water_rhf.json")) 
    
    # for i in 1:5
    #   DF_time = @elapsed begin @time begin 
    #     df_scf_results, density_fitted_properties = full_rhf(denity_fitted_input_file)
    #   end end
    #   RHF_time = @elapsed begin @time begin 
    #     scf_results, properties = full_rhf(input_file)      
    #   end end
    # end
    

    # println("done iwth runs")
    # flush(stdout)

    # println("---------------------------------------------")
    # println("RHF iteration Times (seconds):")
    # print_iteration_times(scf_results["Timings"])
    
    # println("DF-RHF iteration Times (seconds):")
    # print_iteration_times(df_scf_results["Timings"])

  
    # println("---------------------------------------------")

    # println("RHF energy   : $(scf_results["Energy"]), time: $(RHF_time) seconds")
    # println("DF-RHF energy: $(df_scf_results["Energy"]) time: $(DF_time) seconds")

    # Test.@test scf_results["Energy"] ≈ df_scf_results["Energy"] atol=.00015 #15 micro hartree tolerance
    # println("Test run successfully!")
  catch e
    println("check_density_fitted_method_matches_RHF Failed with exception:\n") 
    display(e) 
    flush(stdout)
    exit()
  end 
  JuliaChem.finalize()

end


# check_density_fitted_method_matches_RHF(ARGS[1], ARGS[2])

# df_path = ARGS[1]
# rhf_path = ARGS[2]

df_path = "./example_inputs/density_fitting/water_density_fitted.json"
rhf_path = "./example_inputs/density_fitting/water_rhf.json"

# df_path = "/home/jackson/source/JuliaChem.jl/example_inputs/S22_3/6-31+G_d/benzene_2_water_df.json"
# rhf_path = "/home/jackson/source/JuliaChem.jl/example_inputs/S22_3/6-31+G_d/benzene_2_water.json"

# MP2_Num = "01"
# df_path = joinpath(@__DIR__,  "../example_inputs/density_fitting/$(MP2_Num)_MP2_df.json")
# rhf_path =  joinpath(@__DIR__, "../example_inputs/density_fitting/$(MP2_Num)_MP2.json")

check_density_fitted_method_matches_RHF(df_path, rhf_path)
