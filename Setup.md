# Setting Up and Running JuliaChem for mixed precision 

## Setup Environment 
Get the code 
```sh 
git clone https://github.com/j-hayes/JuliaChem.jl
git checkout feature/mixed_precision
cd JuliaChem.jl
```

```sh 
module load julia/1.11.4 # lets consistently use this version NERSC 
```

Alternatively download Julia from https://julialang.org/downloads/ 
```sh
wget https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.4-linux-x86_64.tar.gz
export PATH=/PATH/TO/Julia/bin:$PATH 
```

``` sh
julia> ]
(@v1.11) pkg> activate mixed_precision_env_perl #activate the julia package environment 
```

Activate the julia package environment for perlmutter 
```
julia> ]
(@v1.11) pkg> ./activate mixed_precision_env_perl 

Activate the environment if on Nova

```sh
(@v1.11) pkg> ./activate mixed_precision_env_nova #activate the julia package environment for nova 
```

Add the local JuliaChem to the environemtn and Build JuliaChem
```sh
(@v1.11) pkg> dev . #adds JuliaChem to the package environment 
(@v1.11) pkg> build JuliaChem # builds JuliaChem C++ Wrapper for LIBINT Integrals
```

## Precompile JuliaChem and Create System Binary 

When running benchmarks one should run using a precompiled binary. This is done using PackageCompiler.jl 

This package uses the normal Julia Precompiler and a script you provide it to run tests execute the code so the whole codebase can be precompiled. 
