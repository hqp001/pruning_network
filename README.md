# Pruning Gurobi MNIST

## How to setup:

### Set up environment:

#### For Bisonnet only:

```bash
module load gurobi-optimizer
pip install onnx onnx2torch vnnlib
```

#### For other Linux machine:

```bash
conda env create -n venv --file=environment.yml
conda activate venv
```

If you install this on Linux computer, the default path for Gurobi license file is `/opt/gurobi/gurobi.lic` or you can change it using:

```bash
conda env config vars set GRB_LICENSE_FILE=<path-to-gurobi.lic>
```

I haven't set it up for Windows so these might not work on Windows

### Set up benchmark:

```bash
sh ./scripts/setup.sh
```

## Folders and Files

- **Solver/**: Contains all solver-related files used in the project.
  - `GurobiSolver.py`: formulate Gurobi Solver
  - `torch2gurobi.py`: convert torch to gurobi 

- **Trainer/**: Contains all trainer-related files used in the project.
  - `Trainer.py`, `Pruner.py`, etc.

- **scripts/**: Contains utility scripts for various tasks.

- **run_gurobi.py**: The main script to run the solver. This script utilizes files from the `Solver` directory.
  
- **training.py**: The main script to run the training process. This script utilizes files from the `Trainer` directory.

- **bisonnet-template_cpu.sh**: Script to run the solver on Bisonnet.
  
- **bisonnet-template_gpu.sh**: Script to run the trainer on Bisonnet.

## How to Run the Scripts

### Running the Trainer

You have to run the training first to train the sparse model

```bash
# Using Python script
python training.py

# or

sbatch bisonnet-template_gpu.sh # on Bisonnet
```


To run the solver, you can either execute the `run_gurobi.py` script or use the `bisonnet-template_cpu.sh` script in the main directory. 

```bash
# Using Python script
python run_gurobi.py

# Using shell script
sh ./bisonnet-template_cpu.sh

# or

sbatch bisonnet-template_cpu.sh # on Bisonnet
```
