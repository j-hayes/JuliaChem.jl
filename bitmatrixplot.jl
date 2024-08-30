using HDF5
using Plots
file = h5open("/home/jackson/source/JuliaChem.jl/testoutputs/C20H42/for_hackathon_hdf5/basis_function_screen_matrix.h5")

matrix = zeros(Bool, 510, 510)
for i in 1:510
    for j in 1:510
        matrix[i,j] = file["data/values"][i,j]
    end
end

total = sum(matrix)
println("total = $total")
screened = 510^2 - total
println("screened = $screened")

heatmap(matrix, aspect_ratio=1, c=:grays, colorbar=false, yflip=true, title="Screening Matrix for C20H42")
savefig("C20H42_screening_matrix.png")