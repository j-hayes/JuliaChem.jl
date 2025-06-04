# Plans 

## Proof Of Concept 


- BLAS Float32 in Dense CPU algorithm 
  - Benchmark with 
      - [ ] S22
        - [ ] DF-RHF Mixed with Float32
        - [ ] vs RHF
        - [ ] vs DF-RHF Float64
        - [ ] vs DF-RHF Mixed Algorithm with 64 Bit
        - [ ] Basis sets
            - [ ] 6-31G++(2p,2d)
            - [ ] CC-PVDZ
            - [ ] CC-PVTZ
      - [ ] Analyze results 
        - [ ] Plot ΔE for S22 comparisons 
        - [ ] Plot ΔE for subset of Inputs for different number of Auxilliary range counts and fock build times 
          - [ ] S22_7 1,2,4,8, Q÷8, Q÷4, Q÷2, Q÷1
          - [ ] S22_4 1,2,4,8, Q÷8, Q÷4, Q÷2, Q÷1
        - [ ] Plot Fock build times S22 vs DF-RHF Float64 / speedup 
  - [ ] Add K and J Symmetry To Dense Algorithm 
- Screened CPU algorithm 
  - Update calculate B algorithm to allow multiple Q ranges on a single MPI rank 
  - Update algorithm to distribute threads to Q ranges and calculate each range on a thread? 
  - Proof of concept, do ranges in memory transfer to setup ranges 