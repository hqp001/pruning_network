#!/bin/bash
#SBATCH -p short # partition (queue)
#SBATCH -N 1 # (leave at 1 unless using multi-node specific code)
#SBATCH -n 8 # number of cores
#SBATCH --mem=32768 # total memory
#SBATCH --job-name="myjob" # job name
#SBATCH -o ./log/slurm.%j.stdout.txt # STDOUT
#SBATCH -e ./log/slurm.%j.stderr.txt # STDERR
#SBATCH --mail-user=username@bucknell.edu # address to email
#SBATCH --mail-type=ALL # mail events (NONE, BEGIN, END, FAIL, ALL)

SEED=73

python run_gurobi.py --seed $SEED --sparsity 0.98
python run_gurobi.py --seed $SEED --sparsity 0.97
python run_gurobi.py --seed $SEED --sparsity 0.96
python run_gurobi.py --seed $SEED --sparsity 0.95




