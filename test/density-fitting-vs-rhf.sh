#!/bin/bash
run_test() {
    echo "Running test $1"
    local S22_number=$1
    local number_of_samples=$2
    local number_of_threads=$3
    local run_name=$4
    local DF_V_RHF="/home/jackson/source/JuliaChem.jl/testoutputs/DF-VS-RHF_S22/$run_name/"
    mkdir -p $DF_V_RHF
    local data_extension=".data"
    local log_extension=".log"
    local data_path="$DF_V_RHF$1$data_extension"
    local log_path="$DF_V_RHF$1$log_extension"

    local script_path=/home/jackson/source/JuliaChem.jl/test/density-fitting-vs-rhf.jl
    
    julia --threads $number_of_threads $script_path $S22_number $data_path $number_of_samples &> $log_path
    echo "finished test $1"
}

# run_test 1 5 hcore 1 df_hcore_vs_rhf_same_etol
run_name=df_vs_rhf_same_etol_3
number_of_threads=16

for i in {1..22}
do
    run_test $i 5 $number_of_threads $run_name
done