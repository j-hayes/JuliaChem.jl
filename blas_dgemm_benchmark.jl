using LinearAlgebra.BLAS

# using MKL
# using BLISBLAS

function do_dgemm(n, iters)
    A = rand(n, n)
    B = rand(n, n)
    C = zeros(n, n)
    time = 0.0
    for i = 1:iters
        time += @elapsed BLAS.gemm!('N', 'N', 1.0, A, B, 1.0, C)
    end
    return time / iters
end

function do_dgemv(n, iters)
    A = rand(n, n)
    x = rand(n)
    y = zeros(n)
    time = 0.0
    for i = 1:iters
        time += @elapsed BLAS.gemv!('N', 1.0, A, x, 1.0, y)
    end
    return time / iters
end

function benchmark_dgemm(n, iters, library)
    time = do_dgemm(n, iters)
    println("Average time using $library with $(BLAS.get_num_threads()) theads for $iters iterations of $n x $n dgemm: $time seconds ")
end

function benchmark_dgemv(n, iters, library)
    time = do_dgemv(n, iters)
    println("Average time using $library with $(BLAS.get_num_threads()) theads for $iters iterations of $n x $n dgemv: $time seconds ")
end

#get n from first arg or default to 1000
n = tryparse(Int, ARGS[1])
if isnothing(n)
    n = 1000
end
blas_threads = tryparse(Int, ARGS[2])
if isnothing(blas_threads)
end

library = "openblas"
if length(ARGS) >=3 
    if ARGS[3] == "mkl"
        using MKL
        library = "mkl"
        # blas info

    elseif ARGS[3] == "blis"
        using BLISBLAS
        library = "blis"
    end
end


#get iters from args or default to 10
iters = tryparse(Int, ARGS[4])
if isnothing(iters)
    iters = 10
end


println(BLAS.get_config())

benchmark_dgemm(n, iters, library)
benchmark_dgemv(n, iters, library)

