function print_normalization_needed_indecies_3d()
    am_matrix = Array{String}(undef, (n_df, n, n))

    for s1 in 1:auxillary_basis_shells_count
        shell_1 = basis_sets.auxillary.shells[s1]
        shell_1_nbasis = basis_sets.auxillary.shells[s1].nbas
        bf_1_pos = basis_sets.auxillary.shells[s1].pos
        for s1b in bf_1_pos:bf_1_pos+shell_1_nbasis-1
            for s2 in 1:basis_shells_count
                shell_2 = basis_sets.primary.shells[s2]
                shell_2_nbasis = basis_sets.primary.shells[s2].nbas
                bf_2_pos = basis_sets.primary.shells[s2].pos
                for s2b in bf_2_pos:bf_2_pos+shell_2_nbasis-1
                    for s3 in 1:basis_shells_count
                        shell_3 = basis_sets.primary.shells[s3]
                        shell_3_nbasis = basis_sets.primary.shells[s3].nbas
                        bf_3_pos = basis_sets.primary.shells[s3].pos
                        for s3b in bf_3_pos:bf_3_pos+shell_3_nbasis-1

                            if shell_1.am > 2 || shell_2.am > 2 || shell_3.am > 2
                                println("index: ($s1b, $s2b, $s3b) am: $(shell_1.am), $(shell_2.am), $(shell_3.am)")
                            end
                            am_matrix[s1b, s2b, s3b] = "$(shell_1.am), $(shell_2.am), $(shell_3.am)"
                        end
                    end
                end

            end
        end
    end
end

function print_normalization_needed_indecies_2d()
    am_matrix = Array{String}(undef, (n_df, n_df))

    for s1 in 1:auxillary_basis_shells_count
        shell_1 = basis_sets.auxillary.shells[s1]
        shell_1_nbasis = basis_sets.auxillary.shells[s1].nbas
        bf_1_pos = basis_sets.auxillary.shells[s1].pos
        for s1b in bf_1_pos:bf_1_pos+shell_1_nbasis-1
            for s2 in 1:s1
                shell_2 = basis_sets.auxillary.shells[s2]
                shell_2_nbasis = basis_sets.auxillary.shells[s2].nbas
                bf_2_pos = basis_sets.auxillary.shells[s2].pos
                for s2b in bf_2_pos:bf_2_pos+shell_2_nbasis-1
                    if shell_1.am > 2 || shell_2.am > 2
                        println("index: ($s1b, $s2b) am: $(shell_1.am), $(shell_2.am)")
                    end
                    am_matrix[s1b, s2b] = "$(shell_1.am), $(shell_2.am),"
                end
            end
        end
    end


    display(am_matrix)
    exit()
end


function check_differentValues(Zxy_Matrix, ZxyCopy)
    size_of = size(Zxy_Matrix)
    size_ofa = size(ZxyCopy)
    for i in 1:size_of[1]
      for j in 1:size_of[2]
        for k in 1:size_of[3]
          delta = abs(Zxy_Matrix[i,j,k] - ZxyCopy[i,j,k])
          if delta > 0.0 
            println("$i,$j,$k: $(ZxyCopy[i,j,k]) => $(Zxy_Matrix[i,j,k])")
          end
        end
      end
    end
  end

function adsafd()
    
  open("/home/jackson/source/JuliaChem.jl/Zxy_from_libint.txt", "r") do io
    for i in 1:n_df
      for j in 1:n
        for k in 1:n
          Zxy_Matrix[i,j,k] = parse(Float64, readline(io))
        end 
      end 
    end 
  end
end

#==
 shell_2 = basis_sets.auxillary.shells[shell_2_index]
      shell_2_basis_count = basis_sets.auxillary.shells[shell_2_index].nbas
      bf_2_pos = basis_sets.auxillary.shells[shell_2_index].pos

      vector = JERI.compute_two_center_eri_block(engine, eri_block_2_center, shell_1_index-1, shell_2_index-1, shell_1_basis_count, shell_2_basis_count)     
      # index1_start = Int64(shell2bf[shell_1_index]) + 1
      # index2_start = Int64(shell2bf[shell_2_index]) + 1            
      eri_i_range = bf_1_pos:bf_1_pos+shell_1_basis_count-1
      eri_j_range = bf_2_pos:bf_2_pos+shell_2_basis_count-1

      # vector_index = 1 
      # for i in eri_i_range 
      #   for j in eri_j_range 
      #     eri_block_2_center_matrix[i,j] = vector[vector_index]
      #     vector_index += 1
      #   end  
      # end
      eri_block_2_center_matrix[eri_i_range, eri_j_range] = reshape(vector, (shell_1_basis_count,shell_2_basis_count))
     
      # axial_normalization_factor(eri_block_2_center_matrix, shell_1, shell_2, shell_1_basis_count, shell_2_basis_count, bf_1_pos, bf_2_pos)
      # display(eri_block_2_center_matrix)
      # println("")

==#