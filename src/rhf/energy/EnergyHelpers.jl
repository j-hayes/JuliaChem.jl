using Base.Threads



function compute_enuc(mol::Molecule)
  E_nuc = 0.0
  for iatom in 1:length(mol), jatom in 1:(iatom-1)
    ix = mol[iatom].atom_center[1] 
    jx = mol[jatom].atom_center[1] 

    iy = mol[iatom].atom_center[2] 
    jy = mol[jatom].atom_center[2] 

    iz = mol[iatom].atom_center[3]
    jz = mol[jatom].atom_center[3]
  
    distance = √((jx-ix)^2 + (jy-iy)^2 + (jz-iz)^2) 
    
    E_nuc += mol[iatom].atom_id*mol[jatom].atom_id/distance
  end 
  
  return E_nuc
end
 
function compute_overlap(S::Matrix{Float64}, basis::Basis,
  jeri_oei_engine)

  for ash in 1:length(basis), bsh in 1:ash
    abas = basis[ash].nbas
    bbas = basis[bsh].nbas
    
    apos = basis[ash].pos
    bpos = basis[bsh].pos
       
    S_block_JERI = zeros(Float64,(abas*bbas,))
    JERI.compute_overlap_block(jeri_oei_engine, S_block_JERI, ash, bsh, 
      length(S_block_JERI))
    axial_normalization_factor(S_block_JERI, basis[ash], basis[bsh])

    idx = 1
    for ibas in 0:abas-1, jbas in 0:bbas-1
      iorb = apos + ibas
      jorb = bpos + jbas
      
      S[max(iorb,jorb),min(iorb,jorb)] = S_block_JERI[idx]
      
      idx += 1 
    end
  end

  for iorb in 1:basis.norb, jorb in 1:iorb
    if iorb != jorb
      S[min(iorb,jorb),max(iorb,jorb)] = S[max(iorb,jorb),min(iorb,jorb)]
    end
  end
end

function compute_ke(T::Matrix{Float64}, basis::Basis, 
  jeri_oei_engine)

  for ash in 1:length(basis), bsh in 1:ash
    abas = basis[ash].nbas
    bbas = basis[bsh].nbas
    
    apos = basis[ash].pos
    bpos = basis[bsh].pos
       
    T_block_JERI = zeros(Float64,(abas*bbas,))
    JERI.compute_kinetic_block(jeri_oei_engine, T_block_JERI, ash, bsh, 
      length(T_block_JERI))
    axial_normalization_factor(T_block_JERI, basis[ash], 
      basis[bsh])

    idx = 1
    for ibas in 0:abas-1, jbas in 0:bbas-1
      iorb = apos + ibas
      jorb = bpos + jbas
      
      T[max(iorb,jorb),min(iorb,jorb)] = T_block_JERI[idx]
      
      idx += 1 
    end
  end
  
  for iorb in 1:basis.norb, jorb in 1:iorb
    if iorb != jorb
      T[min(iorb,jorb),max(iorb,jorb)] = T[max(iorb,jorb),min(iorb,jorb)]
    end
  end
end

function compute_nah(V::Matrix{Float64}, mol::Molecule, 
  basis::Basis, jeri_oei_engine)
  
  #== define ncenter ==#
  #=
  ncenter::Int64 = length(mol)
  
  Z = Vector{Float64}([])
  x = Vector{Float64}([])
  y = Vector{Float64}([])
  z = Vector{Float64}([])

  for atom in mol 
    push!(Z, convert(Float64,atom.atom_id))  
    push!(x, atom.atom_center[1])  
    push!(y, atom.atom_center[2])  
    push!(z, atom.atom_center[3])  
  end
  =#
  for ash in 1:length(basis), bsh in 1:ash
    abas = basis[ash].nbas
    bbas = basis[bsh].nbas
    
    apos = basis[ash].pos
    bpos = basis[bsh].pos
       
    V_block_JERI = zeros(Float64,(abas*bbas,))
    JERI.compute_nuc_attr_block(jeri_oei_engine, V_block_JERI, ash, bsh, 
      length(V_block_JERI))
    axial_normalization_factor(V_block_JERI, basis[ash], 
      basis[bsh])
  
    idx = 1
    for ibas in 0:abas-1, jbas in 0:bbas-1
      iorb = apos + ibas
      jorb = bpos + jbas
      
      V[max(iorb,jorb),min(iorb,jorb)] = V_block_JERI[idx]
      
      idx += 1 
    end
  end
  
  for iorb in 1:basis.norb, jorb in 1:iorb
    if iorb != jorb
      V[min(iorb,jorb),max(iorb,jorb)] = V[max(iorb,jorb),min(iorb,jorb)]
    end
  end
end

function sad_guess(mol::Molecule, basis::Basis)
  basis_symbol = basis.model

  sad_guess = zeros(Float64, (basis.norb, basis.norb))
  h5open(joinpath(@__DIR__, "../../../records/sadgss.h5"),"r") do sadgss 
    anchor = 1
    for atom in mol
      atom_symbol = atom.symbol
     
      sadgss_buf = read(sadgss["$atom_symbol/$basis_symbol"])
      #println("$anchor, $atom")
      #display(sadgss_buf); println()

      sqrt_nbas_guess = trunc(Int,sqrt(length(sadgss_buf)))

      sadgss_idx = 1
      for i in anchor:(anchor+sqrt_nbas_guess-1) 
        for j in anchor:(anchor+sqrt_nbas_guess-1)
          sad_guess[i,j] = sadgss_buf[sadgss_idx]
          sadgss_idx += 1 
        end  
      end
      anchor += sqrt_nbas_guess
    end  
  end
 
  #display(sad_guess) 
  return sad_guess  
end

function compute_schwarz_bounds(schwarz_bounds::Matrix{Float64}, 
  basis::Basis, nsh::Int64)

  max_am = max_ang_mom(basis) 
  eri_quartet_batch = Vector{Float64}(undef,eri_quartet_batch_size(max_am))
  jeri_schwarz_engine = JERI.RHFTEIEngine(basis.basis_cxx, basis.shpdata_cxx)

  for ash in 1:nsh, bsh in 1:ash
    #fill!(eri_quartet_batch, 0.0)
    abas = basis[ash].nbas
    bbas = basis[bsh].nbas
    abshp = triangular_index(ash, bsh)
 
    JERI.compute_eri_block(jeri_schwarz_engine, eri_quartet_batch, 
      ash, bsh, ash, bsh, abshp, abshp, abas*bbas, abas*bbas)
 
    #= axial normalization =#
    axial_normalization_factor(eri_quartet_batch, basis[ash], basis[bsh], 
      basis[ash], basis[bsh], abas, bbas, abas, bbas)

    #== compute schwarz bound ==#
    max_index = BLAS.iamax(abas*bbas*abas*bbas, 
      eri_quartet_batch, 1)
    schwarz_bounds[ash, bsh] = sqrt(abs(eri_quartet_batch[max_index]))
  end

  for ash in 1:nsh, bsh in 1:ash
    if ash != bsh
      schwarz_bounds[min(ash,bsh),max(ash,bsh)] = 
        schwarz_bounds[max(ash,bsh),min(ash,bsh)]
    end
  end
end

#=
"""
		get_oei_matrix(oei::Array{Float64,2})
Summary
======
Extract one-electron integrals from data file object. Kinetic energy integrals,
overlap integrals, and nuclear attraction integrals can all be extracted.

Arguments
======
oei = array of one-electron integrals to extract
"""
=#
function read_in_oei(oei::Vector{T}, nbf::Int) where T
	nbf2 = (nbf*(nbf+1)) >> 1

	oei_matrix = Matrix{Float64}(undef,(nbf,nbf))
	for ibf in 1:nbf2
    i = decompose(ibf)
    j = ibf - triangular_index(i)

		oei_matrix[i,j] = float(oei[ibf])
		oei_matrix[j,i] = oei_matrix[i,j]
	end

	return oei_matrix
end

function DIIS(F::Matrix{Float64}, e_array::Vector{Matrix{Float64}}, 
  F_array::Vector{Matrix{Float64}}, B_dim::Int64)
  
  B = Matrix{Float64}(undef,B_dim+1,B_dim+1)
  for i in 1:B_dim, j in 1:B_dim
    B[i,j] = LinearAlgebra.BLAS.dot(length(e_array[i]), e_array[i], 1, 
      e_array[j], 1)

	  B[i,B_dim+1] = -1.0
	  B[B_dim+1,i] = -1.0
	  B[B_dim+1,B_dim+1] = 0.0
  end
  #DIIS_coeff::Vector{Float64} = [ fill(0.0,B_dim)..., -1.0 ]
  DIIS_coeff::Vector{Float64} = vcat(zeros(B_dim), [-1.0])

  #DIIS_coeff[:], B[:,:], ipiv = LinearAlgebra.LAPACK.gesv!(B, DIIS_coeff)
  DIIS_coeff[:], B[:,:], ipiv = LinearAlgebra.LAPACK.sysv!('U', B, DIIS_coeff)
  
  #fill!(F, zero(Float64))
  LinearAlgebra.BLAS.scal!(length(F), 0.0, F, 1) 
  for index in 1:B_dim
    #F .+= DIIS_coeff[index] .* F_array[index]
    F .+= DIIS_coeff[index] .* F_array[index]
  end
end

function axial_normalization_factor(oei, ash, bsh)
  ama = ash.am
  amb = bsh.am

  na = ash.nbas
  nb = bsh.nbas

  ab = 0 
  for asize::Int64 in 0:(na-1), bsize::Int64 in 0:(nb-1)
    ab += 1 
   
    anorm = axial_norm_fact[asize+1,ama]
    bnorm = axial_norm_fact[bsize+1,amb]
    
    abnorm = anorm*bnorm 
    oei[ab] *= abnorm
  end
end

@inline function axial_normalization_factor(eri_quartet_batch::Vector{Float64},
  μsh::JCModules.Shell, νsh::JCModules.Shell, 
  λsh::JCModules.Shell, σsh::JCModules.Shell,
  nμ::Int, nν::Int, nλ::Int, nσ::Int) 

  #eri_quartet_batch_simint = similar(eri_quartet_batch)
  #println(ish, ",", jsh, ",", ksh, ",", lsh)
  amμ = μsh.am
  amν = νsh.am
  amλ = λsh.am
  amσ = σsh.am

  if amμ < 3 && amν < 3 && amλ < 3 && amσ < 3
    return
  else
    μνλσ = 0 
    for μsize::Int64 in 0:(nμ-1), νsize::Int64 in 0:(nν-1)
      μνλσ = nσ*nλ*νsize + nσ*nλ*nν*μsize
      
      μnorm = get_axial_normalization_factor(μsize+1,amμ)
      νnorm = get_axial_normalization_factor(νsize+1,amν)

      μνnorm = μnorm*νnorm

      for λsize::Int64 in 0:(nλ-1), σsize::Int64 in 0:(nσ-1)
        μνλσ += 1 
   
        λnorm = get_axial_normalization_factor(λsize+1,amλ)
        σnorm = get_axial_normalization_factor(σsize+1,amσ)
    
        λσnorm = λnorm*σnorm 
      
        eri_quartet_batch[μνλσ] *= μνnorm*λσnorm
      end
    end 
  end
end



@inline function axial_normalization_factor_screened!(eri_quartet_batch::Array{Float64},
  μsh::JCModules.Shell, νsh::JCModules.Shell, λsh::JCModules.Shell,
  nμ::Int, nν::Int, nλ::Int,
  μ::Int,ν::Int,λ::Int, sparse_pq_index_map) 

  amμ = μsh.am
  amν = νsh.am
  amλ = λsh.am  
  
  for μsize::Int64 in 0:(nμ-1) 
    μnorm = get_axial_normalization_factor(μsize+1,amμ)
    for νsize::Int64 in 0:(nν-1)
      νnorm = get_axial_normalization_factor(νsize+1,amν)
      μνnorm = μnorm*νnorm
      for λsize::Int64 in 0:(nλ-1) 
        
        screened_index = sparse_pq_index_map[ν+νsize,λ+λsize] 
        if screened_index == 0 || ν+νsize < λ+λsize
          continue
        end
        
        λnorm = get_axial_normalization_factor(λsize+1,amλ)
        normalization_factor = μνnorm*λnorm        
        if amμ < 3 && amν < 3 && amλ < 3 
          normalization_factor = 1.0
        end
        
        eri_quartet_batch[screened_index,μ+μsize] *= normalization_factor # moved AUX to third index
        # if ν+νsize > λ+λsize
        #   inverted_screened_index = sparse_pq_index_map[λ+λsize, ν+νsize] 
        #   eri_quartet_batch[inverted_screened_index,μ+μsize] = eri_quartet_batch[screened_index,μ+μsize]  # moved AUX to third index #this logic is funky to have here for symmetry. This step should be combined with the copy step to be less confusing and more performant
        # end not needed because we are using lower triangle 
      end 
    end
  end 
end

@inline function axial_normalization_factor(eri_quartet_batch::Array{Float64},
  μsh::JCModules.Shell, νsh::JCModules.Shell, λsh::JCModules.Shell,
  nμ::Int, nν::Int, nλ::Int,
  μ::Int,ν::Int,λ::Int) 

  amμ = μsh.am
  amν = νsh.am
  amλ = λsh.am

  
  
  for μsize::Int64 in 0:(nμ-1) 
    μnorm = get_axial_normalization_factor(μsize+1,amμ)
    for νsize::Int64 in 0:(nν-1)
      νnorm = get_axial_normalization_factor(νsize+1,amν)
      μνnorm = μnorm*νnorm
      for λsize::Int64 in 0:(nλ-1)
        λnorm = get_axial_normalization_factor(λsize+1,amλ)
        normalization_factor = μνnorm*λnorm        
        if amμ < 3 && amν < 3 && amλ < 3 
          normalization_factor = 1.0
        end
        eri_quartet_batch[ν+νsize,λ+λsize,μ+μsize] *= normalization_factor # moved AUX to third index
        if ν+νsize > λ+λsize
          eri_quartet_batch[λ+λsize,ν+νsize,μ+μsize] =  eri_quartet_batch[ν+νsize,λ+λsize,μ+μsize]  # moved AUX to third index #this logic is funky to have here for symmetry. This step should be combined with the copy step to be less confusing and more performant
        end
      end 
    end
  end 
end

@inline function axial_normalization_factor(eri_quartet_batch::Array{Float64},
  μsh::JCModules.Shell, νsh::JCModules.Shell,
  nμ::Int, nν::Int,
  μ::Int,ν::Int) 

  amμ = μsh.am
  amν = νsh.am
  
  if amμ < 3 && amν < 3
    return
  end
  for μsize::Int64 in 0:(nμ-1) 
    for νsize::Int64 in 0:(nν-1)
      μnorm = get_axial_normalization_factor(μsize+1,amμ)
      νnorm = get_axial_normalization_factor(νsize+1,amν)
      μνnorm = μnorm*νnorm       
      eri_quartet_batch[μ+μsize,ν+νsize] *= μνnorm      
    end
  end 
end


function eri_quartet_batch_size(max_am)
  return am_to_nbas_cart(max_am)^4
end
