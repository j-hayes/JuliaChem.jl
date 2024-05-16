using CUDA
using CUDA.CUBLAS
using CUDA.CUSOLVER
using LinearAlgebra

function df_rhf_fock_build_GPU!(scf_data, jeri_engine_thread_df::Vector{T}, jeri_engine_thread::Vector{T2},
    basis_sets::CalculationBasisSets,
    occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine,T2<:RHFTEIEngine}
    comm = MPI.COMM_WORLD
    pq = scf_data.μ^2

    AA = scf_data.A
    μμ = scf_data.μ
    ii = scf_data.occ

    CUDA.device!(0)
    
    if iteration == 1
        println("starting GPU DF build")

        two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
        three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options)
        cu_three_center_integrals = CUDA.CuArray{Float64}(undef, ( scf_data.μ, scf_data.μ,scf_data.A))
        cu_J_AB_invt = CUDA.CuArray{Float64}(undef, (scf_data.A, scf_data.A))
        scf_data.D = CUDA.CuArray{Float64}(undef, (scf_data.μ, scf_data.μ,scf_data.A))

        copyto!(cu_J_AB_invt, two_center_integrals)
        copyto!(cu_three_center_integrals, three_center_integrals)

        CUSOLVER.potrf!('L', cu_J_AB_invt)
        CUSOLVER.trtri!('L', 'N', cu_J_AB_invt)

        CUBLAS.gemm!('N', 'T', 1.0, reshape(cu_three_center_integrals, (pq,scf_data.A)), cu_J_AB_invt, 0.0, reshape(scf_data.D, (pq,AA)))


        #clear the memory 
        two_center_integrals = nothing
        cu_three_center_integrals = nothing
        cu_J_AB_invt = nothing        

        #initialize memory for fock build 
        scf_data.occupied_orbital_coefficients = CUDA.CuArray{Float64}(undef, (scf_data.μ, scf_data.occ))
        scf_data.density = CUDA.CuArray{Float64}(undef, (scf_data.μ, scf_data.μ))
        scf_data.coulomb_intermediate = CUDA.CuArray{Float64}(undef, (scf_data.A))
        scf_data.D_tilde = CUDA.CuArray{Float64}(undef, (scf_data.μ,  scf_data.occ, scf_data.A))
        scf_data.two_electron_fock_GPU = CUDA.CuArray{Float64}(undef, (scf_data.μ, scf_data.μ))

    end
    copyto!(scf_data.occupied_orbital_coefficients, occupied_orbital_coefficients)
    CUBLAS.gemm!('N', 'T', 1.0, scf_data.occupied_orbital_coefficients, scf_data.occupied_orbital_coefficients, 0.0, scf_data.density)

    CUBLAS.gemv!('T', 1.0, reshape(scf_data.D, (pq, scf_data.A)), reshape(scf_data.density, pq), 0.0, scf_data.coulomb_intermediate)
    CUBLAS.gemv!('N', 2.0, reshape(scf_data.D, (pq, scf_data.A)), scf_data.coulomb_intermediate , 0.0, reshape(scf_data.two_electron_fock_GPU, pq))

    copyto!(scf_data.two_electron_fock, scf_data.two_electron_fock_GPU)

    CUBLAS.gemm!('T', 'N' , 1.0, reshape(scf_data.D, (μμ, μμ*AA)), scf_data.occupied_orbital_coefficients, 0.0, reshape(scf_data.D_tilde, (μμ*AA,ii)))
    CUBLAS.gemm!('N', 'T', -1.0, reshape(scf_data.D_tilde, (μμ, ii*AA)), reshape(scf_data.D_tilde, (μμ, ii*AA)), 1.0, scf_data.two_electron_fock_GPU)
    
    #copy back the fock matrix to the host
    copyto!(scf_data.two_electron_fock, scf_data.two_electron_fock_GPU)





end
