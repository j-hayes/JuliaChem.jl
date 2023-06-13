using MPI
include("./density-fitting-vs-rhf.jl")
function run()
    # MPI.Init()
    # comm = MPI.COMM_WORLD
    # rank = MPI.Comm_rank(comm)
    # size = MPI.Comm_size(comm)
    # println("rank = $rank size = $size")
    run_df_vs_rhf_test()

end
run()
