#=============================#
#== put needed modules here ==#
#=============================#
# ENV["MKL_DYNAMIC"] = false
# using MKL
import JuliaChem
import Test
using JuliaChem.Shared
using MPI
using LinearAlgebra
using Base.Threads
using ThreadPinning
using CUDA  
include("../example_scripts/full-rhf-repl.jl")


#==================================================================
 Script to check if the Density fitted method 
 values are close to the ones produced by non density fitted RHF
==================================================================#

function check_density_fitted_method_matches_RHF(denity_fitted_input_file, input_file)
  try 

    outputval = 1

    #startup compilation runs
   # screened_cpu_time = @elapsed begin @time begin 
    #   screened_scf_results, screened_properties = full_rhf(input_file)    # screened cpu df-rhf
    # end end
    # df_scf_results, density_fitted_properties = full_rhf(joinpath(@__DIR__, "../example_inputs/density_fitting/water_density_fitted_gpu.json"), output=outputval)
    # df_scf_results, density_fitted_properties = full_rhf(joinpath(@__DIR__, "../example_inputs/density_fitting/water_density_fitted_gpu.json"), output=outputval)
    # cpu_scf_results, cpu_properties = full_rhf(joinpath(@__DIR__, "../example_inputs/density_fitting/water_density_fitted_cpu.json"), output=1)



    for i in 1:1
     
      # sleep(.1)
      df_scf_results, density_fitted_properties = full_rhf(denity_fitted_input_file, output=outputval)
      # df_scf_results, density_fitted_properties = full_rhf(denity_fitted_input_file)
    end
    # DF_time = @elapsed begin @time begin 
    #   df_scf_results, density_fitted_properties = full_rhf(denity_fitted_input_file)
    # end end
    # RHF_time = @elapsed begin @time begin 
    #   scf_results, properties = full_rhf(input_file)      
    # end end

    

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
println(BLAS.get_config())
println("number of gpus =  $(length(CUDA.devices()))")
JuliaChem.initialize() 

n_threads = Threads.nthreads()

comm_rank = MPI.Comm_rank(MPI.COMM_WORLD)
# println("starting JC on rank $comm_rank with $n_threads threads")
# if comm_rank %2 == 0
#   ThreadPinning.pinthreads(0:n_threads-1)
# else
#   ThreadPinning.pinthreads(n_threads:(n_threads*2)-1)
# end
ThreadPinning.pinthreads(0:63)
BLAS.set_num_threads(64)
# check_density_fitted_method_matches_RHF(ARGS[1], ARGS[2])

# df_path = ARGS[1]
# rhf_path = ARGS[2]

# df_path = joinpath(@__DIR__,  "../example_inputs/density_fitting/C20H42_df.json")
# rhf_path = joinpath(@__DIR__,  "../example_inputs/density_fitting/C20H42.json")

# df_path = joinpath(@__DIR__,  "../example_inputs/density_fitting/C40H82_df.json")
# rhf_path = joinpath(@__DIR__,  "../example_inputs/density_fitting/C40H82.json")

# df_path = "/home/jackson/source/JuliaChem.jl/example_inputs/S22_3/6-31+G_d/ammonia_trimer_df.json"
# rhf_path = "/home/jackson/source/JuliaChem.jl/example_inputs/S22_3/6-31+G_d/benzene_2_water.json"

MP2_Num = "03"
df_path = joinpath(@__DIR__,  "../example_inputs/density_fitting/$(MP2_Num)_MP2_df.json")
rhf_path =  joinpath(@__DIR__, "../example_inputs/density_fitting/$(MP2_Num)_MP2.json")

check_density_fitted_method_matches_RHF(df_path, rhf_path)


