using CUDA
using CUDA.CUBLAS
using LinearAlgebra

function df_rhf_fock_build_GPU!(scf_data, jeri_engine_thread_df::Vector{T}, jeri_engine_thread::Vector{T2},
    basis_sets::CalculationBasisSets,
    occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine,T2<:RHFTEIEngine}
    comm = MPI.COMM_WORLD
    pq = scf_data.μ^2

    AA = scf_data.A
    μμ = scf_data.μ
    ii = scf_data.occ
    
    if iteration == 1
        println("starting GPU DF build")

        two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
        three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options)
        three_center_integrals = permutedims(three_center_integrals, [3, 1, 2]) # todo put these in the right order from the start
        cu_three_center_integrals = CUDA.CuArray{Float64}(undef, (scf_data.A, pq))
        cu_J_AB_invt = CUDA.CuArray{Float64}(undef, (scf_data.A, scf_data.A))
        scf_data.D = CUDA.CuArray{Float64}(undef, (scf_data.A,pq))

        copyto!(cu_J_AB_invt, two_center_integrals)
        copyto!(scf_data.D, three_center_integrals)

        CUDA.CUSOLVER.potrf!('L', cu_J_AB_invt)
        CUDA.CUSOLVER.trtri!('L', 'N', cu_J_AB_invt)

        CUDA.CUBLAS.trmm!('L', 'L', 'N', 'N', 1.0, cu_J_AB_invt, scf_data.D, scf_data.D)


        #clear the memory 
        two_center_integrals = nothing
        cu_three_center_integrals = nothing
        cu_J_AB_invt = nothing        

        #initialize memory for fock build 
        scf_data.occupied_orbital_coefficients = CUDA.CuArray{Float64}(undef, (scf_data.μ, scf_data.occ))
        scf_data.density = CUDA.CuArray{Float64}(undef, (scf_data.μ, scf_data.μ))
        scf_data.coulomb_intermediate = CUDA.CuArray{Float64}(undef, (scf_data.A))
        scf_data.D_tilde = CUDA.CuArray{Float64}(undef, (scf_data.occ, scf_data.A, scf_data.μ))
        scf_data.two_electron_fock_GPU = CUDA.CuArray{Float64}(undef, (scf_data.μ, scf_data.μ))

    end
    copyto!(scf_data.occupied_orbital_coefficients, occupied_orbital_coefficients)
    CUBLAS.gemm!('N', 'T', 1.0, scf_data.occupied_orbital_coefficients, scf_data.occupied_orbital_coefficients, 0.0, scf_data.density)

    CUBLAS.gemv!('N', 1.0, scf_data.D, reshape(scf_data.density, pq), 0.0, scf_data.coulomb_intermediate)
    CUBLAS.gemv!('T', 2.0, scf_data.D, scf_data.coulomb_intermediate , 0.0, reshape(scf_data.two_electron_fock_GPU, pq))

    CUBLAS.gemm!('N', 'N' , 1.0, reshape(scf_data.D, (AA*μμ, μμ)), scf_data.occupied_orbital_coefficients, 0.0, reshape(scf_data.D_tilde, (AA*μμ,ii)))
    CUBLAS.gemm!('N', 'T', -1.0, reshape(scf_data.D_tilde, (μμ, AA*ii)), reshape(scf_data.D_tilde, (μμ, AA*ii)), 1.0, scf_data.two_electron_fock_GPU)
    
    #copy back the fock matrix to the host
    copyto!(scf_data.two_electron_fock, scf_data.two_electron_fock_GPU)





end
