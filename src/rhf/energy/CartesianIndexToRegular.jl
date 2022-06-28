a = Array{String, 3}(undef, (4,2,2))

@inline function triangular_index(a::Int,b::Int)
    return (muladd(a,a,-a) >> 1) + b
  end
  
  @inline function triangular_index(a::Int)
    return muladd(a,a,-a) >> 1
  end
  
  @inline function decompose(input::Int)
    #return ceil(Int,(-1.0+√(1+8*input))/2.0)
    return Base.fptosi(
      Int, 
      Base.ceil_llvm(
        0.5*( 
          Base.Math.sqrt_llvm(
            Base.sitofp(Float64, muladd(8,input,1))
          ) - 1.0
        )
      )
    )
  end

function get_value_from_index(array, index)
    indexes = eachindex(view(array))
    cart_index = indexes[index]
    dimensions = size(array)
    i = cart_index[1]
    j  = cart_index[2]
    k  = cart_index[3]
    # i = ((index-1) % dimensions[1])+1
    # j  = (index-1) ÷ dimensions[2]*dimensions[3]
    # k  = 0
    return i,j,k
end

for i in 1:4
    for j in 1:2
        for k in 1:2
            a[i,j,k] = "$i,$j,$k"
        end 
    end
end

a_vec = vec(a)




# for index in 1:4*2*2
#     i,j,k = get_value_from_index(a, index)
    
#     println("$(a_vec[index]) || $i,$j,$k")
# end





# dim = size(a)
# b = CartesianIndices(dim)

# for index in b
#     println("a[$index] = $(a[index])")
# end

# b = Array{Float16, 2}(undef,4,5)

# b[1,3] = 12.32
# b[3,1] = 23234.32
# b[3,3] = 12.32

# if b[1,3] == b[3,1] 
#     println("1,3")
# end


# if b[2,2] != b[3,1] 
#     println("2,2")
# end


# if b[3,3] == b[1,3] 
#     println("1,3")
# end
