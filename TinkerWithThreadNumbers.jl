using Base.Threads
using LinearAlgebra

function main()
    println(BLAS.get_num_threads())
    println(Threads.nthreads())
end

main() 