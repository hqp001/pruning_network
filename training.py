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
from utils.IO import import_model, export_model
from utils.ModelHelpers import double_transform, count_params, init_weights

parser = argparse.ArgumentParser(description="Model state dictionary checker and saver.")

parser.add_argument('--sparsity', default=0.9, type=float, required=False, help="Sparsity level (between 0 and 1)")

parser.add_argument('--model_path', default="./vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx", type=str, required=False, help="Model path")

parser.add_argument('--sub_folder', default="a", type=str, required=False, help="Path to subfolder")

parser.add_argument('--double', action="store_true", help="Convert to model twice as large")

parser.add_argument('--batch_size', default=64, type=int, required=False, help="Batch size for training and testing")

parser.add_argument('--random_init', action="store_true", help="Start with random network or no (not recommended)")

args = parser.parse_args()

SEED = 70
FOLDER_PATH = os.path.dirname(__file__)
MODEL_PATH=args.model_path

# Input info
SPARSITY = args.sparsity
MODEL_NAME, _ = os.path.splitext(os.path.basename(MODEL_PATH))
CATEGORY = os.path.basename(os.path.dirname(os.path.dirname(MODEL_PATH)))

# Output info
SAVE_MODEL_PATH = f"{FOLDER_PATH}/pytorch_model/{args.sub_folder}_{SPARSITY}"
SPARSE_PATH = f"{SAVE_MODEL_PATH}/{MODEL_NAME}_{SPARSITY}.onnx"

# Training info
DOUBLE = args.double
RANDOM_INIT = args.random_init
ROUNDS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = args.batch_size

if CATEGORY == "oval21" or CATEGORY == "sri_resnet_a" or CATEGORY == "cifar2020":

    DATASET = CIFAR10Dataset
    INPUT_SIZE = 3072
    OUTPUT_SIZE = 10
    INPUT_SHAPE = (1, 3, 32, 32)

elif CATEGORY == "mnist_fc":

    DATASET = MNISTDataset
    INPUT_SIZE = 784
    OUTPUT_SIZE = 10
    INPUT_SHAPE = (1, 28, 28)

else:

    raise ValueError(f"Unknown Category: {CATEGORY}")

def set_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def train_model(nn_model):

    print("Device to train: ", DEVICE)

    train_loader = DATASET(train=True, batch_size=BATCH_SIZE).get_data()
    test_loader = DATASET(train=False, batch_size=BATCH_SIZE).get_data()

    if RANDOM_INIT:
        print("Not using trained network")
        nn_model.apply(initialize_weights)

    if DOUBLE:
        assert CATEGORY == "mnist_fc"
        print("Doubling Model")
        nn_model = double_transform(nn_model)
        print(nn_model)

    nn_model = nn_model.to(device=DEVICE)

    trainer = ModelTrainer(max_epochs=1, learning_rate= 1e-2, device=DEVICE)

    print("Accuracy before training: ", trainer.calculate_score(nn_model, test_loader))

    initial_weights = deepcopy(nn_model.state_dict())
    total_parameters = count_params(nn_model)
    prune_pc_per_round = 1 - (1 - SPARSITY) ** (1 / ROUNDS)

    print("Total Params:", total_parameters)

    for round in range(ROUNDS):
        print(f"\nPruning round {round} of {ROUNDS}")

        # Fit the model to the training data
        trainer.train(nn_model, train_loader)

        pruned_model = Pruner(sparsity=prune_pc_per_round).prune(nn_model)

        # Reset model
        init_weights(nn_model, initial_weights)

        # print(f"Model accuracy: {accuracy:.3f}%")
        # print(f"New parameters: {n_pruned_parameters}/{total_parameters}")

    trainer = ModelTrainer(max_epochs=1, learning_rate=1e-3, device=DEVICE)
    trainer.train(nn_model, train_loader)

    Pruner.apply_mask(nn_model)

    test_score = trainer.calculate_score(nn_model, test_loader)

    nn_model = nn_model.to("cpu")
    return nn_model, test_score

def initialize_weights(layer):
    if isinstance(layer, torch.nn.Linear):
        # Initialize weights with a normal distribution
        torch.nn.init.normal_(layer.weight, mean=0.0, std=0.05)
        if layer.bias is not None:
            torch.nn.init.normal_(layer.bias, mean=0.0, std=0.05)


set_seed(SEED)

start = time.time()

dense_model = import_model(MODEL_PATH)

sparse_model = None

sparse_model, validation_score = train_model(import_model(MODEL_PATH))

print("Dense Model: ", dense_model)

print("Sparse Model: ", sparse_model)

os.makedirs(os.path.dirname(SPARSE_PATH), exist_ok=True)

export_model(sparse_model, INPUT_SHAPE, SPARSE_PATH)

print("\n----------------------\n")
print("Test score: ", validation_score)
print("Dense # Parameters: ", count_params(dense_model))
print("Sparse # Parameters: ", count_params(sparse_model))
print("Total time training: ", time.time() - start)
