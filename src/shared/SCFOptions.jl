using JuliaChem.Shared.Constants.SCF_Keywords
mutable struct SCFOptions 
    density_fitting :: Bool 
    contraction_mode :: String
    load :: String 
    guess :: String
    energy_convergence :: Float64
    density_convergence :: Float64
    df_energy_convergence :: Float64
    df_density_convergence :: Float64
    max_iterations :: Int64
    df_max_iterations :: Int64
    df_exchange_block_width :: Int64
    df_screening_sigma :: Float64
end 

function create_default_scf_options()
    return SCFOptions(
        false, # density_fitting
        SCF_Keywords.ContractionMode.default,
        SCF_Keywords.Load.default,
        SCF_Keywords.Guess.default,
        SCF_Keywords.Convergence.energy_default,
        SCF_Keywords.Convergence.density_default,
        SCF_Keywords.Convergence.energy_default,
        SCF_Keywords.Convergence.density_default,
        SCF_Keywords.Convergence.max_iterations_default,
        SCF_Keywords.Convergence.df_max_iterations_default,
        SCF_Keywords.Screening.df_exchange_block_width_default,
        SCF_Keywords.Screening.df_sigma_default
        )
end

function create_scf_options(scf_flags)


    guess::String = haskey(scf_flags, Guess.guess) ?
    scf_flags[Guess.guess] : Guess.default


    load::String = haskey(scf_flags, IntegralLoad.load) ? 
        scf_flags[IntegralLoad.load] : IntegralLoad.default

    contraction_mode::String = haskey(scf_flags, ContractionMode.contraction_mode) ?
        scf_flags[ContractionMode.contraction_mode] : ContractionMode.default

    do_density_fitting::Bool = haskey(scf_flags, SCFType.scf_type) ? 
        lowercase(scf_flags[SCFType.scf_type]) == SCFType.density_fitting : false

    energy_convergence = haskey(scf_flags, Convergence.energy) ?
        scf_flags[Convergence.energy] : Convergence.energy_default

    density_convergence = haskey(scf_flags, Convergence.density) ?
        scf_flags[Convergence.density] : Convergence.density_default

    df_energy_convergence = haskey(scf_flags, Convergence.density_fitting_energy) ?
        scf_flags[Convergence.density_fitting_energy] : Convergence.energy_default

    df_density_convergence = haskey(scf_flags, Convergence.density_fitting_density) ?
        scf_flags[Convergence.density_fitting_density] : Convergence.density_default
  
    niter::Int = haskey(scf_flags, Convergence.max_iterations) ?
        scf_flags[Convergence.max_iterations] : Convergence.max_iterations_default

    df_exchange_block_width::Int = haskey(scf_flags, Screening.df_exchange_block_width) ? 
        scf_flags[Screening.df_exchange_block_width] : Screening.df_exchange_block_width_default 
    df_screening_sigma::Float64 = haskey(scf_flags, Screening.df_sigma) ? 
        scf_flags[Screening.df_sigma] : Screening.df_sigma_default

    df_niter = 0
    if do_density_fitting 
        df_niter = niter        
        
    elseif guess == Guess.density_fitting
        df_niter::Int = haskey(scf_flags, Convergence.df_max_iterations) ?
        scf_flags[Convergence.df_max_iterations] : Convergence.df_max_iterations_default        
    end

    return SCFOptions(
        do_density_fitting,
        contraction_mode,
        load,
        guess,
        energy_convergence,
        density_convergence,
        df_energy_convergence,
        df_density_convergence,
        niter,
        df_niter, 
        df_exchange_block_width,
        df_screening_sigma)
end

function print_scf_options(options::SCFOptions)
    println("SCF Options:")
    println("------------------------------")
   
    
    println("Integral Load: ", options.load)
    
    println("Guess: ", options.guess)
    println("Energy Convergence: ", options.energy_convergence)
    println("Density Convergence: ", options.density_convergence)
    println("Max Iterations: ", options.max_iterations)
    println("--------------------------------")
    if options.density_fitting
        println("")
        println("Density Fitting Options:")
        println("--------------------------------")
        println("Contraction Mode: ", options.contraction_mode)
        println("DF Energy Convergence: ", options.df_energy_convergence)
        println("DF Density Convergence: ", options.df_density_convergence)
        println("DF Exchange Block Width: nbas ÷ ", options.df_exchange_block_width)
        println("DF Screening Sigma: ", options.df_screening_sigma)
        println("--------------------------------")
    end
end

export SCFOptions, create_default_scf_options, create_scf_options, print_scf_options