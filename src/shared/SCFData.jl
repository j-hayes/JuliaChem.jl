mutable struct ScreeningData
    sparse_pq_index_map
    basis_function_screen_matrix
    sparse_p_start_indices
    non_screened_p_indices_count
    non_zero_coefficients # GTFOCK paper equation 4 "In order to use optimized matrix multiplication library functions to compute W(i, Q), for each p, we need a dense matrix consisting of the nonzero rows of C(r, i)."
    screened_triangular_indicies
    shell_screen_matrix
    screened_triangular_indicies_to_2d
    r_ranges #todo rename
    B_ranges #todo rename
    triangular_indices_count :: Int
    screened_indices_count :: Int
end

mutable struct SCFData
    D 
    D_triangle
    D_tilde 
    two_electron_fock_triangle 
    two_electron_fock
    two_electron_fock_GPU
    thread_two_electron_fock 
    coulomb_intermediate
    density
    occupied_orbital_coefficients
    density_array
    J
    K
    k_blocks
    screening_data :: ScreeningData
    μ :: Int
    occ :: Int
    A :: Int
end


function SCFData()
    sd = ScreeningData([],[], [], [], [], [], [], [], [], [], 0, 0)
    return SCFData([], [], [], [], [],[], [], [], [], [],[] ,[],[],[], sd, 0, 0, 0)
end

export SCFData