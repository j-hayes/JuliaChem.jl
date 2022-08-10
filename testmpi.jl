using MPI

function do_stuff()
    MPI.Init()
    MPI.Finalize()
    println("stuff")
end

do_stuff()