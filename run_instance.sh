#!/bin/bash

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 v1 benchmark_identifier onnx_file_path vnnlib_file_path results_file_path timeout"
    exit 1
fi

# Assign arguments to variables
v1=$1
benchmark_identifier=$2
onnx_file_path=$3
vnnlib_file_path=$4
results_file_path=$5
timeout=$6

# Run the Python script with the provided arguments
python ./run_gurobi.py --sparsity 0 --subfolder "sparse_std" --model_path "$onnx_file_path" --instance_path "$vnnlib_file_path" --output_path "$results_file_path" --time_limit "$timeout"
