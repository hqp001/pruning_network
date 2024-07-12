# run measurements for all categories for a single tool (passed on command line)
# seven args: 'v1' (version string), tool_scripts_folder, vnncomp_folder, result_csv_file, counterexamples_folder, categories, all|different|first
#
# for example ./run_all_categories.sh v1 ~/repositories/simple_adversarial_generator/vnncomp_scripts . ./out.csv ./counterexamples "test acasxu" all


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_FOLDER="${SCRIPT_DIR}/.."
BENCHMARK_FOLDER="${MAIN_FOLDER}/vnncomp2022_benchmarks"
RESULTS_FOLDER="${MAIN_FOLDER}/vnncomp2022_results"

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <sparsity> <subfolder> <callback>"
    exit 1
fi


SPARSITY=$1
SUBFOLDER=$2
CALLBACK=$3

FOLDER_NAME="${SUBFOLDER}_${SPARSITY}_${CALLBACK}"
FOLDER_PATH="${RESULTS_FOLDER}/${FOLDER_NAME}"
CATEGORY="mnist_fc"

export SPARSITY
export SUBFOLDER
export CALLBACK
export FOLDER_PATH

if [ -d "$FOLDER_PATH" ]; then
    echo "Error: The directory '$FOLDER_PATH' already exists."
    rm -r $FOLDER_PATH || exit
fi

mkdir "$FOLDER_PATH"
if [ $? -eq 0 ]; then
    echo "Directory '$FOLDER_PATH' created successfully."
else
    echo "Failed to create directory '$FOLDER_PATH'. Check permissions or directory path."
fi

ITEMS_TO_COPY=(
    "run_gurobi.py"
    "pytorch_model/${SUBFOLDER}_${SPARSITY}"
    "Solver"
    "scripts/run_instance.sh"
    "scripts/prepare_instance.sh"
    "utils"
)

for SOURCE_ITEM in "${ITEMS_TO_COPY[@]}"; do

  if [ -e "$SOURCE_ITEM" ]; then

    SOURCE_DIR=$(dirname "$SOURCE_ITEM")
    DESTINATION_DIR="${FOLDER_PATH}/${SOURCE_DIR}"
    echo $DESTINATION_DIR
    mkdir -p "$DESTINATION_DIR"

    cp -r "${MAIN_FOLDER}/${SOURCE_ITEM}" "$DESTINATION_DIR" || exit

    if [ $? -eq 0 ]; then
      echo "Copied: $SOURCE_ITEM"
    else
      echo "Error copying: $SOURCE_ITEM"
    fi

  else
    echo "Source item does not exist: $SOURCE_ITEM"
  fi

done

cd ${FOLDER_PATH} || exit

${BENCHMARK_FOLDER}/run_all_categories.sh v1 "${FOLDER_PATH}/scripts" ${BENCHMARK_FOLDER} ${FOLDER_PATH}/results.csv ${FOLDER_PATH} $CATEGORY first

for SOURCE_ITEM in "${ITEMS_TO_COPY[@]}"; do

    SOURCE_DIR=$(dirname "$SOURCE_ITEM")
    DESTINATION_ITEM="${FOLDER_PATH}/${SOURCE_ITEM}"

    if [ -e "$DESTINATION_ITEM" ]; then

        rm -r "$DESTINATION_ITEM"

        if [ $? -eq 0 ]; then
            echo "Deleted: $DESTINATION_ITEM"
        else
            echo "Error deleting: $DESTINATION_ITEM"
        fi

    else
        echo "Destination item does not exist: $DESTINATION_ITEM"
    fi

    find "${FOLDER_PATH}/$SOURCE_DIR" -type d -empty -delete

done



