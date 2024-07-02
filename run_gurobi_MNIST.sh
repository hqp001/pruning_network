#!/bin/bash
#SBATCH -p short # partition (queue)
#SBATCH -N 1 # (leave at 1 unless using multi-node specific code)
#SBATCH -n 16 # number of cores
#SBATCH --mem=16384 # total memory
#SBATCH --job-name="myjob" # job name
#SBATCH -o ./log/slurm.%j.stdout.txt # STDOUT
#SBATCH -e ./log/slurm.%j.stderr.txt # STDERR
#SBATCH --mail-user=username@bucknell.edu # address to email
#SBATCH --mail-type=ALL # mail events (NONE, BEGIN, END, FAIL, ALL)

# run measurements for all categories for a single tool (passed on command line)
# seven args: 'v1' (version string), tool_scripts_folder, vnncomp_folder, result_csv_file, counterexamples_folder, categories, all|different|first
#
# for example ./run_all_categories.sh v1 ~/repositories/simple_adversarial_generator/vnncomp_scripts . ./out.csv ./counterexamples "test acasxu" all


if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <sparsity> <subfolder>"
    exit 1
fi


SPARSITY=$1
SUBFOLDER=$2
CALLBACK="dense_passing"

export SPARSITY
export SUBFOLDER
export CALLBACK

FOLDER_NAME="${SUBFOLDER}_${SPARSITY}"
FOLDER_PATH="./vnncomp2022_results/${FOLDER_NAME}"
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

./vnncomp2022_benchmarks/run_all_categories.sh v1 . ./vnncomp2022_benchmarks ./${FOLDER_PATH}/results.csv ./${FOLDER_PATH} $CATEGORY all




