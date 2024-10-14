using JuliaChem.JCModules

using JuliaChem.Shared
using JuliaChem.Shared.JCTC

function set_scf_options_data!(timing::JCTiming,scf_options::SCFOptions)
    timing.options[JCTC.density_fitting] = string(scf_options.density_fitting)
    timing.options[JCTC.contraction_mode] = scf_options.contraction_mode
    timing.options[JCTC.load] = scf_options.load
    timing.options[JCTC.guess] = scf_options.guess
    timing.options[JCTC.energy_convergence] = string(scf_options.energy_convergence)
    timing.options[JCTC.density_convergence] = string(scf_options.density_convergence)
    timing.options[JCTC.df_energy_convergence] = string(scf_options.df_energy_convergence)
    timing.options[JCTC.df_density_convergence] = string(scf_options.df_density_convergence)
    timing.options[JCTC.max_iterations] = string(scf_options.max_iterations)
    timing.options[JCTC.df_max_iterations] = string(scf_options.df_max_iterations)
    timing.options[JCTC.df_exchange_block_width] = string(scf_options.df_exchange_block_width)
    timing.options[JCTC.df_screening_sigma] = string(scf_options.df_screening_sigma)
    timing.options[JCTC.df_screen_exchange] = string(scf_options.df_screen_exchange)

end

function set_threads_and_ranks!(timing::JCTiming, n_threads::Int, n_ranks::Int)
    timing.non_timing_data[JCTC.n_threads] = string(n_threads)
    timing.non_timing_data[JCTC.n_ranks] = string(n_ranks)
end

function set_converged!(timing::JCTiming,scf_converged::Bool, iterations::Int, energy::Float64)
    timing[JCTC.converged] = string(scf_converged)
    timing[JCTC.n_iterations] = string(iterations)
    timing.scf_energy = energy
end

function set_basis_info!(jc_timing::JCTiming, basis::Basis, aux_basis::Basis)
    jc_timing.non_timing_data[JCTC.n_basis_functions] = string(basis.norb)
    if !isnothing(aux_basis)
        jc_timing.non_timing_data[JCTC.n_auxiliary_basis_functions] = string(aux_basis.norb)
    else
        jc_timing.non_timing_data[JCTC.n_auxiliary_basis_functions] = "0"
    end
    jc_timing.non_timing_data[JCTC.n_electrons] = string(basis.nels)
    jc_timing.non_timing_data[JCTC.n_occupied_orbitals] = string(basis.nels÷2)
end

function set_converge_properties!(jc_timing, scf_converged, n_iterations, scf_energy)
    jc_timing.converged = scf_converged
    jc_timing.non_timing_data[JCTC.n_iterations] = string(n_iterations)
    jc_timing.scf_energy = scf_energy
end

export set_scf_options_data!, set_threads_and_ranks!, set_converged!, set_basis_info!, set_converge_properties!