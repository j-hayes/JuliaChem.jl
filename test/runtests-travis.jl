import Test

include("../tools/travis/travis-rhf.jl")

#== select input files ==#
directory = joinpath(@__DIR__, "../example_inputs/S22/")
inputs = readdir(directory)
inputs .= directory .* inputs

display(inputs)

#== establish correct values for energies, dipoles, and HOMO-LUMO gaps ==#
GAMESS_energies = Vector{Float64}([
  -112.4047194144,
  -152.0671593526,
  -377.5889420312,
  -337.9245601137,
  -825.0480089998,
  -623.3855095909,
  -916.1413738989,
  -80.4073036256,
  -156.0878533332,
  -270.9302743515,
  -461.4497600413,
  -525.4098675881,
  -825.0271971388,
  -592.2263255700,
  -916.1246231957,
  -154.8750579586,
  -306.7601557517,
  -286.9286565278,
  -323.6144109921,
  -461.4537798340,
  -592.2351527581,
  -611.1928406081
])

GAMESS_dipole_moments = Vector{Float64}([
  1.0E-6, #approximately 0.0        
  2.696653,
  1.0E-6, #approximately 0.0        
  1.0E-6, #approximately 0.0        
  1.0E-6, #approximately 0.0        
  3.239972, 
  1.990013, 
  1.0E-6, #approximately 0.0        
  1.0E-6, #approximately 0.0        
  0.280546, 
  1.0E-6, #approximately 0.0        
  0.227480, 
  5.604885, 
  1.860770, 
  3.478761, 
  0.493345, 
  2.422111, 
  1.770430, 
  4.142949, 
  0.651396, 
  3.194881, 
  3.486793 
])

GAMESS_HOMO_LUMO_gaps = Vector{Float64}([
  0.6192,
  0.6493,
  0.6283,
  0.5964,
  0.4622,
  0.4068,
  0.4054,
  0.7288,
  0.5424,
  0.4746,  
  0.4538,
  0.4372,
  0.4447,
  0.4031,
  0.4154,
  0.5441,
  0.4736,
  0.4742,
  0.4757,
  0.4563,
  0.3869,
  0.4226
])

#== initialize JuliaChem ==#
JuliaChem.initialize()

#== run S22 calculations ==#
molecules = Vector{Int}([ 2, 3 ]) 

s22_test_results = Dict([]) 
for imol in molecules 
  s22_test_results[imol] = travis_rhf(inputs[imol])
end

#== check energies ==#
Test.@testset "S22 Energy" begin
  for imol in molecules 
    Test.@test s22_test_results[imol][:Energy]["Energy"] ≈ GAMESS_energies[imol] 
  end
end

#== check dipole moments ==#
Test.@testset "S22 Dipoles" begin
  for imol in molecules 
    if GAMESS_dipole_moments[imol] == 1.0E-6
      Test.@test abs(s22_test_results[imol][:Properties]["Dipole"][:moment]) <= GAMESS_dipole_moments[imol] #check if approximately zero 
    else
      Test.@test s22_test_results[imol][:Properties]["Dipole"][:moment] ≈ GAMESS_dipole_moments[imol] atol=5.0E-5
    end
  end
end

#== check energies ==#
Test.@testset "S22 HOMO-LUMO Gaps" begin
  for imol in molecules 
    Test.@test s22_test_results[imol][:Properties]["MO Energies"][:homo_lumo] ≈ GAMESS_HOMO_LUMO_gaps[imol] atol=5.0E-4
  end
end

#== finalize JuliaChem ==#
JuliaChem.finalize()       
