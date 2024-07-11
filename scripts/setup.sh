#!/bin/bash

# Set the repository URL
BENCHMARK_URL="https://github.com/ChristopherBrix/vnncomp2022_benchmarks.git"
BENCHMARK_FOLDER="./vnncomp2022_benchmarks"

RESULTS_URL="https://github.com/ChristopherBrix/vnncomp2022_results.git"
RESULTS_FOLDER="./vnncomp2022_results"

install_repo() {
    local repo_url="$1"
    local folder_name="$2"
    if [ -d "$folder_name" ]; then
        echo "The folder '$folder_name' exists."
    else
        echo "The folder '$folder_name' does not exist."
        echo "Cloning the repository..."
        git clone "$repo_url"
        echo "Operation completed."
    fi

}

pip install -r ./requirements.txt
install_repo $BENCHMARK_URL $BENCHMARK_FOLDER
install_repo $RESULTS_URL $RESULTS_FOLDER
gunzip ./vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/*.gz ./vnncomp2022_benchmarks/benchmarks/mnist_fc/vnnlib/*.gz

echo "Complete setup"
