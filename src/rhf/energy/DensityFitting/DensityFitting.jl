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
  df_rfh_fock_build_single_node_GTFOCK!(scf_data, jeri_engine_thread,
  basis_sets, occupied_orbital_coefficients, iteration, scf_options)

  # if comm_size == 1
  #   println("do single node")  
  #   df_rfh_fock_build_single_node!(scf_data, jeri_engine_thread,
  #   basis_sets, occupied_orbital_coefficients, iteration, scf_options)
  # else  
  #   df_rhf_fock_build_multi_node!(scf_data, jeri_engine_thread,
  #   basis_sets, occupied_orbital_coefficients, iteration, scf_options)
  # end
  
end

@inline function df_rfh_fock_build_single_node_GTFOCK!(scf_data, jeri_engine_thread::Vector{T},
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}

  comm = MPI.COMM_WORLD
  if MPI.Comm_size(comm) == 1
    if iteration == 1
      two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread, basis_sets, scf_options)
      three_center_integrals = calculate_three_center_integrals(jeri_engine_thread, basis_sets, scf_options)
      calculate_D!(scf_data, two_center_integrals, three_center_integrals, scf_options)
    end  
    @tensor density[μ, ν] := occupied_orbital_coefficients[μ,i]*occupied_orbital_coefficients[ν,i]
    @tensor scf_data.coulomb_intermediate[A] = scf_data.D[μ,ν,A]*density[μ,ν]
    @tensor scf_data.two_electron_fock[μ, ν] = 2*scf_data.coulomb_intermediate[A]*scf_data.D[μ,ν,A]
    @tensor scf_data.D_tilde[ν,i,A] = scf_data.D[ν, μμ, A] * occupied_orbital_coefficients[μμ, i]
    @tensor scf_data.two_electron_fock[μ, ν] -= scf_data.D_tilde[μ, i, A]*scf_data.D_tilde[ν, i, A]
  else
    if iteration == 1
      @time two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread, basis_sets, scf_options)
      @time three_center_integrals = calculate_three_center_integrals(jeri_engine_thread, basis_sets, scf_options)
      @time calculate_D_multi_node!(scf_data, two_center_integrals, three_center_integrals, basis_sets, scf_options)
    end  
    @time begin 
    scf_data.coulomb_intermediate .= 0.0
    scf_data.D_tilde .= 0.0
    scf_data.two_electron_fock .= 0.0

    @tensor density[μ, ν] := occupied_orbital_coefficients[μ,i]*occupied_orbital_coefficients[ν,i]
    indicies = get_df_static_basis_indices(basis_sets, MPI.Comm_size(comm), MPI.Comm_rank(comm))
    @tensor scf_data.coulomb_intermediate[A] = scf_data.D[μ,ν,A]*density[μ,ν]
    @tensor scf_data.two_electron_fock[μ, ν] = 2.0*scf_data.coulomb_intermediate[A]*scf_data.D[μ,ν,A]
    @tensor scf_data.D_tilde[ν,i,A] = scf_data.D[ν, μμ, A]*occupied_orbital_coefficients[μμ, i]
    @tensor scf_data.two_electron_fock[μ, ν] -= scf_data.D_tilde[μ, i, A]*scf_data.D_tilde[ν, i, A]
    
    # @sync for node_aux_index in eachindex(indicies)   
    #   Threads.@spawn begin
    #     scf_data.coulomb_intermediate[node_aux_index] = dot(scf_data.D[:,:,node_aux_index], density[:,:])
    #   end
    # end
    # @sync for μ in 1:scf_data.μ
    #   Threads.@spawn begin
    #     for ν in 1:scf_data.μ
    #         scf_data.two_electron_fock[μ,ν] = 2*dot(scf_data.D[μ, ν,:] , scf_data.coulomb_intermediate[:])            
    #     end            
    #   end
    # end
    

    # @sync for node_aux_index in eachindex(indicies)   
    #   Threads.@spawn begin
    #     BLAS.gemm!('N', 'N', 1.0, view(scf_data.D, :,:,node_aux_index), occupied_orbital_coefficients, 0.0, view(scf_data.D_tilde, :,:,node_aux_index))
    #   end
    # end

    # @sync for μ in 1:scf_data.μ
    #   Threads.@spawn begin
    #     for ν in 1:scf_data.μ
    #       scf_data.two_electron_fock[μ, ν] -= dot(view(scf_data.D_tilde, μ,:,:), view(scf_data.D_tilde, ν,:,:))
    #     end
    #   end
    # end

    MPI.Barrier(comm)
    MPI.Allreduce!(scf_data.two_electron_fock, MPI.SUM, comm)
    MPI.Barrier(comm)

  end
  end # end time
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
  calculate_D_tilde!(scf_data, occupied_orbital_coefficients,  scf_options)
  calculate_coulomb!(scf_data, occupied_orbital_coefficients,  scf_options) 
  calculate_exchange!(scf_data,scf_options)
end

@inline function df_rhf_fock_build_multi_node!(scf_data, jeri_engine_thread::Vector{T},
  basis_sets::CalculationBasisSets,
  occupied_orbital_coefficients, iteration, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}
  comm = MPI.COMM_WORLD

  if iteration == 1
     two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread, basis_sets, scf_options)
     three_center_integrals = calculate_three_center_integrals(jeri_engine_thread, basis_sets, scf_options)
     calculate_D_multi_node!(scf_data, two_center_integrals, three_center_integrals, basis_sets ,scf_options)
  end
  scf_data.D_tilde .= 0.0
  scf_data.two_electron_fock .= 0.0

  calculate_D_tilde_multi_node!(scf_data, occupied_orbital_coefficients, basis_sets, scf_options)
  calculate_coulomb_multi_node!(scf_data, occupied_orbital_coefficients, basis_sets, scf_options)
  calculate_exchange_multi_node!(scf_data, scf_options)

end

#original tensor operation  @tensor two_electron_fock[μ, ν] -= xiK[μ, νν, AA] * xiK[ν, νν, AA]
function calculate_exchange!(scf_data, scf_options::SCFOptions)
  @tensor scf_data.two_electron_fock[μ, ν] -= scf_data.D_tilde[μ, ii, AA] * scf_data.D_tilde[ν,ii, AA]
end

@inline function calculate_exchange_multi_node!(scf_data, scf_options::SCFOptions)
  comm = MPI.COMM_WORLD
  number_of_elements = scf_data.A * scf_data.occ
  indicies = filter(x-> x[1] >= x[2], CartesianIndices(scf_data.two_electron_fock))
  batch_start, batch_end = get_contraction_batch_bounds(length(indicies))
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
    # @tensoropt (μμ => 10, ii => 10) scf_data.two_electron_fock[μ, ν] := 2.0 * scf_data.D[μ, ν, A] *
    # scf_data.D_tilde[μμ, ii, A] * occupied_orbital_coefficients[μμ, ii]  
    for μ in 1:scf_data.μ
      for ν in 1:scf_data.μ
        for A in 1:scf_data.A
            for ii in 1:scf_data.occ
                for μμ in 1:scf_data.μ
                    scf_data.two_electron_fock[μ, ν] += 2.0 * scf_data.D[μ, ν, A] *
                    scf_data.D_tilde[μμ, ii, A] * occupied_orbital_coefficients[μμ, ii]
                end  
            end
            # scf_data.two_electron_fock[μ, ν] += 2.0 * scf_data.D[μ, ν, A]
        end
      end
    end
end

function calculate_coulomb_multi_node!(scf_data, occupied_orbital_coefficients, basis_sets,  scf_options::SCFOptions)
  comm = MPI.COMM_WORLD
  indicies = get_df_static_basis_indices(basis_sets, MPI.Comm_size(comm), MPI.Comm_rank(comm))
  @sync for A in indicies
    Threads.@spawn begin
      intermediate = 0.0      
      for ii in 1:scf_data.occ
        for μμ in 1:scf_data.μ
          intermediate += scf_data.D_tilde[μμ, ii,A] * occupied_orbital_coefficients[μμ, ii]
        end
      end

      for μ in 1:scf_data.μ
        for ν in 1:scf_data.μ
           scf_data.two_electron_fock[μ,ν] +=  2.0 * scf_data.D[μ,ν,A] #* intermediate
        end
      end
    end
  end    


  MPI.Barrier(comm)
  MPI.Allreduce!(scf_data.two_electron_fock, MPI.SUM, comm)
  MPI.Barrier(comm)
end


# calculate xiK 
@inline function calculate_D_tilde!(scf_data, occupied_orbital_coefficients, scf_options::SCFOptions)
  @tensor scf_data.D_tilde[ν,i,A] = scf_data.D[ν, μμ, A] * occupied_orbital_coefficients[μμ, i]
end

# calculate xiK 
@inline function calculate_D_tilde_multi_node!(scf_data, occupied_orbital_coefficients, basis_sets, scf_options::SCFOptions)
  comm = MPI.COMM_WORLD
  indicies = get_df_static_basis_indices(basis_sets, MPI.Comm_size(comm), MPI.Comm_rank(comm))

  @sync for A in indicies
    Threads.@spawn begin
      for i in 1:scf_data.occ
        for ν in 1:scf_data.μ
          scf_data.D_tilde[ν,i,A] = BLAS.dot(scf_data.occ, view(scf_data.D, ν,:,A), 1, view(occupied_orbital_coefficients, :,i), 1)
        end
      end
    end
  end

  MPI.Barrier(comm)
  MPI.Allreduce!(scf_data.D, MPI.SUM, comm)
  MPI.Barrier(comm)
  if MPI.Comm_rank(comm) == 0
    for index in CartesianIndices(scf_data.D)
      println("scf_data.D[$(index[1]), $(index[2]), $(index[3])] = $(scf_data.D[index]))]")
    end
  end


end


@inline function calculate_D!(scf_data, two_center_integrals, three_center_integrals, scf_options::SCFOptions)
  comm = MPI.COMM_WORLD
    J_AB_invt = convert(Array, transpose(cholesky(Hermitian(two_center_integrals, :L)).L \ I))
    @tensor scf_data.D[μ, ν, A] = three_center_integrals[μ, ν, BB] * J_AB_invt[BB, A]
end # end function calculate_D

function calculate_D_multi_node!(scf_data, two_center_integrals, three_center_integrals, basis_sets, scf_options::SCFOptions)
  comm = MPI.COMM_WORLD
  # this needs to be mpi parallelized? or is it okay to just do it on every node?
  J_AB_invt = convert(Array, transpose(cholesky(Hermitian(two_center_integrals, :L)).L \ I))
  indicies = get_df_static_basis_indices(basis_sets, MPI.Comm_size(comm), MPI.Comm_rank(comm))
  @tensor scf_data.D[μ, ν, A] = three_center_integrals[μ, ν, BB] * J_AB_invt[:, indicies][BB,A]
end

function get_contraction_batch_bounds(total_number_of_operations)
  comm = MPI.COMM_WORLD

  batch_size = convert(Int, ceil(total_number_of_operations / MPI.Comm_size(comm)))
  batch_start = batch_size * MPI.Comm_rank(comm) + 1
  batch_end = min(batch_start + batch_size - 1, total_number_of_operations)

  return batch_start, batch_end
end