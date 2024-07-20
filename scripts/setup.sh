#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath "$0"))
BENCHMARK_FOLDER="$SCRIPT_DIR/../vnncomp2022_benchmarks"
RESULTS_FOLDER="$SCRIPT_DIR/../vnncomp2022_results"

# Set the repository URL
BENCHMARK_URL="https://github.com/ChristopherBrix/vnncomp2022_benchmarks.git"

RESULTS_URL="https://github.com/ChristopherBrix/vnncomp2022_results.git"

install_repo() {
    local repo_url="$1"
    local folder_name="$2"
    if [ -d "$folder_name" ]; then
        echo "The folder '$folder_name' exists."
    else
        echo "The folder '$folder_name' does not exist."
        echo "Cloning the repository..."
        git clone "$repo_url" --depth 1
        echo "Operation completed."
    fi

}

install_repo $BENCHMARK_URL $BENCHMARK_FOLDER
install_repo $RESULTS_URL $RESULTS_FOLDER

echo "Finish downloading benchmarks"

echo "Unzipping files. Please wait..."

gunzip ${BENCHMARK_FOLDER}/benchmarks/*/onnx/* ${BENCHMARK_FOLDER}/benchmarks/*/vnnlib/*

echo "Complete setup"
