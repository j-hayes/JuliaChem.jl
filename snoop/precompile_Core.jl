function _precompile_core()
    println("Precompiling core...")
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(Core.Compiler.getindex), Tuple{typeof(Base.has_offset_axes), Bool}, Int64})
    precompile(Tuple{typeof(Core.Compiler.getindex), Tuple{typeof(Base.max), Int64}, Int64})
    precompile(Tuple{typeof(Core.Compiler.getindex), Tuple{Float64, Bool}, Int64})
    precompile(Tuple{typeof(Core.Compiler.getindex), Tuple{Float64, Float64}, Int64})
    precompile(Tuple{typeof(Core.Compiler.getindex), Tuple{typeof(Base.complex)}, Int64})
    precompile(Tuple{typeof(Core.Compiler.getindex), Tuple{typeof(Base.parse), DataType}, Int64})
    precompile(Tuple{typeof(Core.Compiler.getindex), Tuple{typeof(Base.sqrt)}, Int64})
    precompile(Tuple{typeof(Core.Compiler.getindex), Tuple{typeof(Base.float)}, Int64})
    precompile(Tuple{typeof(Core.Compiler._typename), DataType})
    precompile(Tuple{typeof(Core.Compiler.getindex), Tuple{typeof(Base.adjoint)}, Int64})
    precompile(Tuple{typeof(Core.Compiler.getindex), Tuple{typeof(Base.conj)}, Int64})
    precompile(Tuple{typeof(Core.Compiler.getindex), Tuple{typeof(Base.:(^))}, Int64})
    precompile(Tuple{typeof(Core.Compiler.getindex), Tuple{Float64, Int64}, Int64})
    precompile(Tuple{typeof(Core.Compiler.getindex), Tuple{typeof(Base.adjoint), Int64}, Int64})
end
