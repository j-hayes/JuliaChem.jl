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
    sparse_pq_index_map
    basis_function_screen_matrix
    density_array
    J
    μ :: Int
    occ :: Int
    A :: Int
end

function SCFData()
   
    return SCFData([], [], [], [], [],[], [], [], [],[] ,[],[],[],[] , 0, 0, 0)
end

export SCFData