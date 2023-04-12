using Base.Threads
using LinearAlgebra
using TensorOperations
using JuliaChem.Shared.Constants.SCF_Keywords
using JuliaChem.Shared
#== Density Fitted Restricted Hartree-Fock, Fock build step ==#
#== 
indices for all tensor contractions 
  duplicated letters: (e.g.) dd dummy variables that will be summed Overlap
  A = Auxillary Basis orbital
  μ,ν = Primary Basis orbital
  i = occupied orbitals
==#

@inline function df_rhf_fock_build!(scf_data, jeri_engine_thread::Vector{T},
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}


  comm = MPI.COMM_WORLD

  comm_size = MPI.Comm_size(comm)

  if comm_size == 1 
    df_rfh_fock_build_single_node!(scf_data, jeri_engine_thread,
    basis_sets, occupied_orbital_coefficients, iteration, scf_options)

  else
    
    df_rhf_fock_build_multi_node!(scf_data, jeri_engine_thread,
    basis_sets, occupied_orbital_coefficients, iteration, scf_options)
  end
  
end

@inline function df_rfh_fock_build_single_node!(scf_data, jeri_engine_thread::Vector{T},
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}

  comm = MPI.COMM_WORLD
  if iteration == 1
    two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread, basis_sets, scf_options)
    three_center_integrals = calculate_three_center_integrals(jeri_engine_thread, basis_sets, scf_options)
    calculate_D!(scf_data, two_center_integrals, three_center_integrals, scf_options)
  end
  calculate_D_tilde!(scf_data, occupied_orbital_coefficients, scf_options)
  calculate_coulomb!(scf_data, occupied_orbital_coefficients, scf_options) 
  calculate_exchange!(scf_data,scf_options)
end

@inline function df_rhf_fock_build_multi_node!(scf_data, jeri_engine_thread::Vector{T},
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}
  comm = MPI.COMM_WORLD

  if iteration == 1
    @time two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread, basis_sets, scf_options)
    @time three_center_integrals = calculate_three_center_integrals(jeri_engine_thread, basis_sets, scf_options)
    @time calculate_D_multi_node!(scf_data, two_center_integrals, three_center_integrals, scf_options)
  end
  scf_data.D_tilde .= 0.0
  scf_data.two_electron_fock .= 0.0

  @time calculate_D_tilde_multi_node!(scf_data, occupied_orbital_coefficients, scf_options)
  @time calculate_coulomb_multi_node!(scf_data, occupied_orbital_coefficients, scf_options)
  @time calculate_exchange_multi_node!(scf_data, scf_options)

end

#original tensor operation  @tensor two_electron_fock[μ, ν] -= xiK[μ, νν, AA] * xiK[ν, νν, AA]
function calculate_exchange!(scf_data, scf_options::SCFOptions)
  @tensor scf_data.two_electron_fock[μ, ν] -= scf_data.D_tilde[νν, AA, μ] * scf_data.D_tilde[νν, AA, ν]
end

@inline function calculate_exchange_multi_node!(scf_data, scf_options::SCFOptions)
  comm = MPI.COMM_WORLD
  number_of_elements = scf_data.A * scf_data.occ
  indicies = filter(x-> x[1] >= x[2], CartesianIndices(scf_data.two_electron_fock))
  batch_start, batch_end = get_contraction_batch_bounds(length(indicies))
  println("batch_start $(batch_start) batch_end $(batch_end) total: $(length(indicies))")
  @sync for index in batch_start:batch_end
    Threads.@spawn begin
        ν = indicies[index][1]
        μ = indicies[index][2]
        exchange = BLAS.dot(number_of_elements, view(scf_data.D_tilde, :, :,μ), 1 ,view(scf_data.D_tilde,:, :,ν), 1)
        scf_data.two_electron_fock[ν, μ] -= exchange
        if ν != μ
          scf_data.two_electron_fock[μ, ν] -= exchange
        end      
    end
  end
  MPI.Barrier(comm)
  MPI.Allreduce!(scf_data.two_electron_fock, MPI.SUM, comm)
  MPI.Barrier(comm)

end

@inline function calculate_coulomb!(scf_data, occupied_orbital_coefficients,
  scf_options::SCFOptions)
    @tensoropt (μμ => 10, ii => 10) scf_data.two_electron_fock[μ, ν] := 2.0 * scf_data.D[ν, A,μ] *
    scf_data.D_tilde[A, ii,μμ] * occupied_orbital_coefficients[μμ, ii]    
end

function calculate_coulomb_multi_node!(scf_data, occupied_orbital_coefficients, scf_options::SCFOptions)
  comm = MPI.COMM_WORLD
  for A in 1:scf_data.A
    scf_data.coulomb_intermediate[A] = dot(transpose(view(scf_data.D_tilde, A,:, :)), occupied_orbital_coefficients)
  end

  batch_start, batch_end = get_contraction_batch_bounds(scf_data.μ)
  @sync for μ in batch_start:batch_end
    Threads.@spawn begin
      # mul!(view(scf_data.two_electron_fock, :,μ), view(scf_data.D, :, :,μ), scf_data.coulomb_intermediate, 2.0, 0.0)
      BLAS.gemm!('N', 'N', 2.0, view(scf_data.D, :, :,μ), scf_data.coulomb_intermediate, 0.0, view(scf_data.two_electron_fock, :,μ))
    end
  end    
end


# calculate xiK 
@inline function calculate_D_tilde!(scf_data, occupied_orbital_coefficients, scf_options::SCFOptions)
   @tensor scf_data.D_tilde[A, i,ν] = scf_data.D[ν, A, μμ] * occupied_orbital_coefficients[μμ, i]
end

# calculate xiK 
@inline function calculate_D_tilde_multi_node!(scf_data, occupied_orbital_coefficients, scf_options::SCFOptions)
  comm = MPI.COMM_WORLD
  batch_start, batch_end = get_contraction_batch_bounds(scf_data.μ)

  @sync for μ in batch_start:batch_end
    Threads.@spawn begin
      BLAS.gemm!('N', 'N', 1.0, scf_data.D[μ, :, :], occupied_orbital_coefficients, 0.0,  view(scf_data.D_tilde, :, :, μ))
    end
  end

  MPI.Barrier(comm)
  MPI.Allreduce!(scf_data.D_tilde, MPI.SUM, comm)
  MPI.Barrier(comm)
end


@inline function calculate_D!(scf_data, two_center_integrals, three_center_integrals, scf_options::SCFOptions)
  comm = MPI.COMM_WORLD
  # this needs to be mpi parallelized
    J_AB_invt = convert(Array, transpose(cholesky(Hermitian(two_center_integrals, :L)).L \ I))
    @tensor scf_data.D[ν, A,μ] = three_center_integrals[μ, ν, BB] * J_AB_invt[BB, A]
end # end function calculate_D



function calculate_D_multi_node!(scf_data, two_center_integrals, three_center_integrals, scf_options::SCFOptions)
  comm = MPI.COMM_WORLD
  # this needs to be mpi parallelized? or is it okay to just do it on every node?
  J_AB_invt = convert(Array, transpose(cholesky(Hermitian(two_center_integrals, :L)).L \ I))
  batch_start, batch_end = get_contraction_batch_bounds(scf_data.μ)
  @sync for μ in batch_start:batch_end
    Threads.@spawn begin
      BLAS.gemm!('N', 'N', 1.0, three_center_integrals[μ, :, :], J_AB_invt, 0.0, view(scf_data.D, :, :, μ))
    end
  end

  MPI.Barrier(comm)
  MPI.Allreduce!(scf_data.D, MPI.SUM, comm)
  MPI.Barrier(comm)
end

function get_contraction_batch_bounds(total_number_of_operations)
  comm = MPI.COMM_WORLD

  batch_size = convert(Int, ceil(total_number_of_operations / MPI.Comm_size(comm)))
  batch_start = batch_size * MPI.Comm_rank(comm) + 1
  batch_end = min(batch_start + batch_size - 1, total_number_of_operations)

  return batch_start, batch_end
end