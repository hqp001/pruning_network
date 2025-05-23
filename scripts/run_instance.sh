#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_FILE="$SCRIPT_DIR/../run_gurobi.py"

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

echo "${SUBFOLDER}_${SPARSITY}" $CALLBACK

# Run the Python script with the provided arguments
python ${RUN_FILE} --sparsity $SPARSITY --callback $CALLBACK --sub_folder "${SUBFOLDER}" --model_path "$onnx_file_path" --instance_path "$vnnlib_file_path" --output_path "$results_file_path" --time_limit "$timeout"
