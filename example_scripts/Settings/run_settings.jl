function get_settings(settings_id)
    if settings_id == "DF_RHF_screenedCPU"
        return DF_RHF_screenedCPU()
    elseif settings_id == "RHF_staticCPU"
        return RHF_staticCPU()
    elseif settings_id == "RHF_dynamicCPU"
        return RHF_dynamicCPU()
    elseif settings_id == "DF_RHF_screenedCPU_as_guess"
        return DF_RHF_screenedCPU_as_guess()
    elseif settings_id == "DF_RHF_denseCPU"
        return DF_RHF_denseCPU()
    elseif settings_id == "DF_RHF_GPU_adaptive"
        return DF_RHF_GPU_adaptive()
    elseif settings_id == "DF_RHF_GPU_dense"
        return DF_RHF_GPU_dense()
    elseif settings_id == "DF_RHF_denseCPU_mixed"
        return DF_RHF_denseCPU_mixed()
    else
        throw(ErrorException("Error: settings_id not recognized"))        
    end
end

function DF_RHF_screenedCPU()
    scf_keywords = Dict{String,Any}()
    scf_keywords["scf_type"] = "df"
    # scf_keywords["ndiis"] = 2
    scf_keywords["guess"] = "hcore"
    scf_keywords["dele"] = 1E-4
    scf_keywords["rmsd"] = 1E-4
    scf_keywords["df_sigma"] = 1E-6
    scf_keywords["load"] = "static"
    scf_keywords["contraction_mode"] = "screened"
    scf_keywords["niter"] = 50
    scf_keywords["df_exchange_n_blocks"] = 10
    return scf_keywords
end



function DF_RHF_screenedCPU_as_guess()
    scf_keywords = DF_RHF_screenedCPU()
    scf_keywords["guess"] = "df"
    scf_keywords["df_dele"] = 1E-3
    scf_keywords["df_rmsd"] = 1E-3
    return scf_keywords
end

function DF_RHF_denseCPU()
    scf_keywords = DF_RHF_screenedCPU()
    scf_keywords["contraction_mode"] = "dense"
    return scf_keywords
end

function RHF_staticCPU()
    scf_keywords = Dict{String,Any}()
    scf_keywords["scf_type"] = "rhf"
    scf_keywords["guess"] = "hcore"
    scf_keywords["load"] = "static"
    scf_keywords["dele"] = 1E-4
    scf_keywords["rmsd"] = 1E-4
    scf_keywords["niter"] = 50
    return scf_keywords
end

function RHF_dynamicCPU()
    scf_keywords = RHF_staticCPU()
    scf_keywords["load"] = "dynamic"
    return scf_keywords
end

function DF_RHF_GPU_adaptive()
    scf_keywords = Dict{String,Any}()
    scf_keywords["scf_type"] = "df"
    scf_keywords["guess"] = "hcore"
    scf_keywords["dele"] = 1E-4
    scf_keywords["rmsd"] = 1E-4
    scf_keywords["df_sigma"] = 1E-6
    scf_keywords["load"] = "static"
    scf_keywords["contraction_mode"] = "GPU"
    scf_keywords["niter"] = 50
    scf_keywords["df_exchange_n_blocks"] = 0
    scf_keywords["df_screen_exchange"] = false
    # #GPU algorithm specific
    scf_keywords["df_force_dense"] = false
    scf_keywords["df_use_adaptive"] = true
    scf_keywords["df_K_sym_type"] = "square"
    return scf_keywords
end

function DF_RHF_GPU_dense()
    scf_keywords = Dict{String,Any}()
    scf_keywords["scf_type"] = "df"
    scf_keywords["guess"] = "hcore"
    scf_keywords["dele"] = 1E-4
    scf_keywords["rmsd"] = 1E-4
    scf_keywords["df_sigma"] = 1E-6
    scf_keywords["load"] = "static"
    scf_keywords["contraction_mode"] = "GPU"
    scf_keywords["niter"] = 50
    scf_keywords["df_exchange_n_blocks"] = 0
    scf_keywords["df_screen_exchange"] = false
    # #GPU algorithm specific
    scf_keywords["df_force_dense"] = true
    scf_keywords["df_use_adaptive"] = false
    scf_keywords["df_K_sym_type"] = "square"
    return scf_keywords
end

function DF_RHF_denseCPU_mixed()
    scf_keywords = DF_RHF_denseCPU()
    scf_keywords["do_mixed_precision"] = true
    scf_keywords["contraction_float_type"] = "half"
    return scf_keywords
end