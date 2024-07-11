#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
        git clone "$repo_url"
        echo "Operation completed."
    fi

}

install_repo $BENCHMARK_URL $BENCHMARK_FOLDER
install_repo $RESULTS_URL $RESULTS_FOLDER

${BENCHMARK_FOLDER}/setup.sh

echo "Complete setup"
