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

SPARSITY=0.5

python run_gurobi_onnx.py --sparsity $SPARSITY --num_layers 2
python run_gurobi_onnx.py --sparsity $SPARSITY --num_layers 4
python run_gurobi_onnx.py --sparsity $SPARSITY --num_layers 6


