import argparse
import time
import os
import random
from copy import deepcopy

import torch
import numpy as np
import onnx
from onnx2torch import convert

from Trainer.Dataset import MNISTDataset, CIFAR10Dataset
from Trainer.Pruner import Pruner
from Trainer.Trainer import ModelTrainer
from utils.Model import ONNXModel, DoubleModel

parser = argparse.ArgumentParser(description="Model state dictionary checker and saver.")

parser.add_argument('--sparsity', default=0.5, type=float, required=False, help="Sparsity level (between 0 and 1)")

parser.add_argument('--model_path', default="./vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx", type=str, required=False, help="Model path")

parser.add_argument('--sub_folder', default="a", type=str, required=False, help="Path to subfolder")

args = parser.parse_args()

SEED = 70
SPARSITY = args.sparsity
FOLDER_PATH = os.path.dirname(__file__)
# MODEL_PATH=f"{FOLDER_PATH}/benchmarks/test/test_nano.onnx"
MODEL_PATH=args.model_path


MODEL_NAME, _ = os.path.splitext(os.path.basename(MODEL_PATH))
SAVE_MODEL_PATH = f"{FOLDER_PATH}/pytorch_model/{args.sub_folder}_{SPARSITY}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DENSE_PATH = f"{SAVE_MODEL_PATH}/{MODEL_NAME}.pth"
SPARSE_PATH = f"{SAVE_MODEL_PATH}/{MODEL_NAME}_{SPARSITY}.pth"


TRAIN_FIRST = True
ROUNDS = 1
DOUBLE = False

def set_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def train_model(nn_model):

    print("Device to train: ", DEVICE)

    train_loader = MNISTDataset(train=True, batch_size=64).get_data()
    test_loader = MNISTDataset(train=False, batch_size=64).get_data()

    if not TRAIN_FIRST:
        print("Not using trained network")
        nn_model.apply(initialize_weights)

    if DOUBLE:
        print("Doubling Model")
        nn_model = DoubleModel(nn_model.layers)
        print(nn_model)

    nn_model = nn_model.to(device=DEVICE)

    trainer = ModelTrainer(max_epochs=1, learning_rate= 1e-2, device=DEVICE)

    print("Accuracy before training: ", trainer.calculate_score(nn_model, test_loader))

    initial_weights = deepcopy(nn_model.state_dict())
    total_parameters = nn_model.count_parameters()
    prune_pc_per_round = 1 - (1 - SPARSITY) ** (1 / ROUNDS)

    print("Total Params:", total_parameters)

    for round in range(ROUNDS):
        print(f"\nPruning round {round} of {ROUNDS}")

        # Fit the model to the training data
        trainer.train(nn_model, train_loader)

        pruned_model = Pruner(model=nn_model, sparsity=prune_pc_per_round).prune().get_model()

        # Reset model
        nn_model.initialize_weights(initial_weights)

        # print(f"Model accuracy: {accuracy:.3f}%")
        # print(f"New parameters: {n_pruned_parameters}/{total_parameters}")

    trainer = ModelTrainer(max_epochs=1, learning_rate=1e-3, device=DEVICE)
    trainer.train(nn_model, train_loader)

    nn_model.apply_mask()

    test_score = trainer.calculate_score(nn_model, test_loader)

    nn_model = nn_model.to("cpu")
    return nn_model, test_score

def dense_model_train():

    # Convert to PyTorch
    onnx_model = convert(onnx.load(MODEL_PATH))

    torch_model = ONNXModel(onnx_model)

    return torch_model

def initialize_weights(layer):
    if isinstance(layer, torch.nn.Linear):
        # Initialize weights with a normal distribution
        torch.nn.init.normal_(layer.weight, mean=0.0, std=0.05)
        if layer.bias is not None:
            torch.nn.init.normal_(layer.bias, mean=0.0, std=0.05)


set_seed(SEED)

start = time.time()

dense_model = dense_model_train()

print(dense_model)


if os.path.exists(DENSE_PATH):
    print(f"The file {DENSE_PATH} exists.")

    # Load the state dictionary from the file
    loaded_state_dict = torch.load(DENSE_PATH)

    # Get the current model's state dictionary
    current_state_dict = dense_model.state_dict()

    # Compare the state dictionaries
    state_dicts_equal = all(torch.equal(current_state_dict[key], loaded_state_dict[key]) for key in current_state_dict)

    assert state_dicts_equal == True

else:
        # Save the model state dictionary
        print(f"The file {DENSE_PATH} does not exist. Saving the model state dictionary.")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(DENSE_PATH), exist_ok=True)

        torch.save(dense_model.state_dict(), f'{DENSE_PATH}')

sparse_model = None

sparse_model, validation_score = train_model(dense_model_train())

torch.save(sparse_model.state_dict(), SPARSE_PATH)

print("\n----------------------\n")
print("Test score: ", validation_score)
print("Dense # Parameters: ", dense_model.count_parameters())
print("Sparse # Parameters: ", sparse_model.count_parameters())
print("Total time training: ", time.time() - start)
