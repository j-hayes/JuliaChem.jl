## Stats needed from each run 

- scf energy  
- iteration times
- fock build times 
- J time
- K time 
- number of iterations
- df iteration start
- df iteration end 
- scf options 
    - density_fitting
    - contraction mode 
    - load 
    - df_screening_sigma
    - df_screen_exchange  
- basis info
    - number of basis functions
    - number of aux basis functions 
    - basis name
    - aux basis name 

save to an hdf5 file with this data 

Run 1

S22 
- RHF 
    - dynamic 
- DF-RHF
    - screened
- DF-RHF guess 
    - screened 

screening sigma = 1E-06 

