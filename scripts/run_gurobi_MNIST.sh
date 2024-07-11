# run measurements for all categories for a single tool (passed on command line)
# seven args: 'v1' (version string), tool_scripts_folder, vnncomp_folder, result_csv_file, counterexamples_folder, categories, all|different|first
#
# for example ./run_all_categories.sh v1 ~/repositories/simple_adversarial_generator/vnncomp_scripts . ./out.csv ./counterexamples "test acasxu" all


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_FOLDER="$SCRIPT_DIR/../vnncomp2022_benchmarks"
RESULTS_FOLDER="$SCRIPT_DIR/../vnncomp2022_results"

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <sparsity> <subfolder> <callback>"
    exit 1
fi


SPARSITY=$1
SUBFOLDER=$2
CALLBACK=$3

export SPARSITY
export SUBFOLDER
export CALLBACK

FOLDER_NAME="${SUBFOLDER}_${SPARSITY}_${CALLBACK}"
FOLDER_PATH="${RESULTS_FOLDER}/${FOLDER_NAME}"
CATEGORY="mnist_fc"

if [ -d "$FOLDER_PATH" ]; then
    echo "Error: The directory '$FOLDER_PATH' already exists."
else
    # Attempt to create the directory
    mkdir "$FOLDER_PATH"
    if [ $? -eq 0 ]; then
        echo "Directory '$FOLDER_PATH' created successfully."
    else
        echo "Failed to create directory '$FOLDER_PATH'. Check permissions or directory path."
    fi
fi

${BENCHMARK_FOLDER}/run_all_categories.sh v1 ${SCRIPT_DIR} ${BENCHMARK_FOLDER} ${FOLDER_PATH}/results.csv ${FOLDER_PATH} $CATEGORY all




