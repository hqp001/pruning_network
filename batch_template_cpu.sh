#!/bin/bash
#SBATCH -p gpu # partition (queue)
#SBATCH -N 1 # (leave at 1 unless using multi-node specific code)
#SBATCH -n 2 # number of cores
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --mem=16384 # total memory
#SBATCH --job-name="myjob" # job name
#SBATCH -o ./log/slurm.%j.stdout.txt # STDOUT
#SBATCH -e ./log/slurm.%j.stderr.txt # STDERR
#SBATCH --mail-user=username@bucknell.edu # address to email
#SBATCH --mail-type=ALL # mail events (NONE, BEGIN, END, FAIL, ALL)

SPARSITY=0.95
SUBFOLDER="adam_no_softmax"

python training.py --sparsity $SPARSITY --model_path "./vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx" --sub_folder $SUBFOLDER
python training.py --sparsity $SPARSITY --model_path "./vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x4.onnx" --sub_folder $SUBFOLDER
python training.py --sparsity $SPARSITY --model_path "./vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x6.onnx" --sub_folder $SUBFOLDER

