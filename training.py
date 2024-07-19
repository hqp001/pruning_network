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
from utils.Model import ONNXModel, DoubleModel, count_params, init_weights, apply_mask

parser = argparse.ArgumentParser(description="Model state dictionary checker and saver.")

parser.add_argument('--sparsity', default=0.9, type=float, required=False, help="Sparsity level (between 0 and 1)")

parser.add_argument('--model_path', default="./vnncomp2022_benchmarks/benchmarks/sri_resnet_a/onnx/resnet_3b2_bn_mixup_adv_4.0_bs128_lr-1.onnx", type=str, required=False, help="Model path")

parser.add_argument('--sub_folder', default="a", type=str, required=False, help="Path to subfolder")

parser.add_argument('--double', action="store_true", help="Convert to model twice as large")

args = parser.parse_args()

SEED = 70
SPARSITY = args.sparsity
FOLDER_PATH = os.path.dirname(__file__)
MODEL_PATH=args.model_path


MODEL_NAME, _ = os.path.splitext(os.path.basename(MODEL_PATH))
SAVE_MODEL_PATH = f"{FOLDER_PATH}/pytorch_model/{args.sub_folder}_{SPARSITY}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DENSE_PATH = f"{SAVE_MODEL_PATH}/{MODEL_NAME}.pth"
SPARSE_PATH = f"{SAVE_MODEL_PATH}/{MODEL_NAME}_{SPARSITY}.onnx"


TRAIN_FIRST = True
ROUNDS = 1
DOUBLE = args.double # TODO: Implement todo
CATEGORY = os.path.basename(os.path.dirname(os.path.dirname(MODEL_PATH)))

if CATEGORY == "oval21" or CATEGORY == "sri_resnet_a":

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

    train_loader = DATASET(train=True, batch_size=1).get_data()
    test_loader = DATASET(train=False, batch_size=1).get_data()

    if not TRAIN_FIRST:
        print("Not using trained network")
        nn_model.apply(initialize_weights)

    if DOUBLE:
        assert CATEGORY == "mnist_fc"
        print("Doubling Model")
        nn_model = DoubleModel(nn_model.layers)
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

        pruned_model = Pruner(model=nn_model, sparsity=prune_pc_per_round).prune().get_model()

        # Reset model
        init_weights(nn_model, initial_weights)

        # print(f"Model accuracy: {accuracy:.3f}%")
        # print(f"New parameters: {n_pruned_parameters}/{total_parameters}")

    trainer = ModelTrainer(max_epochs=1, learning_rate=1e-3, device=DEVICE)
    trainer.train(nn_model, train_loader)

    apply_mask(nn_model)

    test_score = trainer.calculate_score(nn_model, test_loader)

    nn_model = nn_model.to("cpu")
    return nn_model, test_score

def import_model(file_path):

    # Convert to PyTorch
    onnx_model = convert(onnx.load(file_path))

    return onnx_model

def export_model(model, file_path):

    dummy_input = torch.randn(*INPUT_SHAPE)

    print(model(dummy_input))

    torch.onnx.export(model,
        dummy_input,
        file_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={ 'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}})

def initialize_weights(layer):
    if isinstance(layer, torch.nn.Linear):
        # Initialize weights with a normal distribution
        torch.nn.init.normal_(layer.weight, mean=0.0, std=0.05)
        if layer.bias is not None:
            torch.nn.init.normal_(layer.bias, mean=0.0, std=0.05)


set_seed(SEED)

start = time.time()

dense_model = import_model(MODEL_PATH)

export_model(dense_model, "./a.onnx")

sparse_model = None

sparse_model, validation_score = train_model(import_model(MODEL_PATH))

print("Dense Model: ", dense_model)

print("Sparse Model: ", sparse_model)

os.makedirs(os.path.dirname(SPARSE_PATH), exist_ok=True)

export_model(sparse_model, SPARSE_PATH)

print("\n----------------------\n")
print("Test score: ", validation_score)
print("Dense # Parameters: ", count_params(dense_model))
print("Sparse # Parameters: ", count_params(sparse_model))
print("Total time training: ", time.time() - start)
