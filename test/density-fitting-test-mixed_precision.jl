#=============================#
#== put needed modules here ==#
#=============================#
ENV["MKL_DYNAMIC"] = false
# using MKL
println("starting density fitting test"); flush(stdout)
using JuliaChem
println("imported JuliaChem"); flush(stdout)
using Test
using JuliaChem.Shared
using JuliaChem.Shared.JCTC

using MPI
using LinearAlgebra
using Base.Threads

include("../example_scripts/full-rhf-repl.jl")
# include("../example_scripts/save_jc_timings.jl")

#==================================================================
 Script to check if the Density fitted method 
 values are close to the ones produced by non density fitted RHF
==================================================================#

function check_density_fitted_method_matches_RHF(denity_fitted_input_file::String, rhf_input_file::String)
  # try 

    
    #warmup
    outputval = 5
    ENV["DO_MIXED"] = "true"
    df_scf_results_mixed, density_fitted_properties_mixed = full_rhf(joinpath(@__DIR__, "../example_inputs/density_fitting/water_density_fitted.json"), output=outputval)
    ENV["DO_MIXED"] = "false"
    df_scf_results, density_fitted_properties = full_rhf(joinpath(@__DIR__, "../example_inputs/density_fitting/water_density_fitted.json"), output=outputval)
    
    println("DF-RHF mixed precision energy   : $(df_scf_results_mixed["Energy"])")
    println("DF-RHF double precision energy   : $(df_scf_results["Energy"])")
   
    energy_diff = abs(df_scf_results["Energy"] - df_scf_results_mixed["Energy"])
    println("energy difference: $energy_diff")
    
    ENV["NUM_Q_RANGES"] = 64
    println("running density fitted file $denity_fitted_input_file")
    ENV["DO_MIXED"] = "true"
    df_scf_results_mixed, density_fitted_properties_mixed = full_rhf(denity_fitted_input_file, output=outputval)
  
    ENV["DO_MIXED"] = "false"
    df_scf_results_double, density_fitted_properties_double = full_rhf(denity_fitted_input_file, output=outputval)
   
    println("DF-RHF mixed precision energy  32 : $(df_scf_results_mixed["Energy"])")
    println("DF-RHF double precision energy   : $(df_scf_results_double["Energy"])")

    BLAS.set_num_threads(1)
    println("running rhf file $rhf_input_file")
    rhf_scf_results, properties_rhf = full_rhf(rhf_input_file, output=outputval)


    println("RHF energy  : $(rhf_scf_results["Energy"])")
    diff_rhf_vs_mixed = abs(rhf_scf_results["Energy"] - df_scf_results_mixed["Energy"])
    diff_rhf_vs_double_precision = abs(rhf_scf_results["Energy"] - df_scf_results_double["Energy"])
    println("RHF vs DF-RHF mixed precision energy difference: $diff_rhf_vs_mixed")
    println("RHF vs DF-RHF double precision energy difference: $diff_rhf_vs_double_precision")

end

function main()
  JuliaChem.initialize() 
  BLAS.set_num_threads(64)
  input_file = "/pscratch/sd/j/jhayes1/source/JuliaChem.jl/example_inputs/density_fitting/C20H42_df.json"
  rhf_input_file = "/pscratch/sd/j/jhayes1/source/JuliaChem.jl/example_inputs/density_fitting/C20H42_rhf.json"
  check_density_fitted_method_matches_RHF(input_file, rhf_input_file)

  
end

main()