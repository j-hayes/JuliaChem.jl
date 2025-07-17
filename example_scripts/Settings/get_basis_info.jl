function get_basis_names(basis_id)
    lower_basis_id = lowercase(basis_id)
    if lower_basis_id == "cc-pvdz-ri"
        return "cc-pVDZ", "cc-pVDZ-RIFIT"
    elseif lower_basis_id == lowercase("6-31G")
        return "6-31G", "cc-pVDZ-JKFIT"
    elseif lower_basis_id == "cc-pvdz-jk"
        return "cc-pVDZ", "cc-pVDZ-JKFIT"
    elseif lower_basis_id == "cc-pvtz-jk"
        return "cc-pVTZ", "cc-pVTZ-JKFIT"
    elseif lower_basis_id == "cc-pvtz-ri"
        return "cc-pVTZ", "cc-pVTZ-RIFIT"
    elseif lower_basis_id == lowercase("6-311++G_2d_2p_")
        return "6-311++G(2d,2p)", "aug-cc-pVTZ-JKFIT"
    elseif lower_basis_id == lowercase("6-31G(d,p)")
        return "6-31G**", "6-31G**-RIFIT"
    end
    error("basis_id not found: $basis_id")

end