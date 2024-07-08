from Dataset import MNISTDataset
import argparse
import numpy as np
import torch
from copy import deepcopy
import random
from Model import ONNXModel
from Pruner import ModelTrainer, Pruner
from onnx2torch import convert
import time
import onnx
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Model state dictionary checker and saver.")

parser.add_argument('--sparsity', default=0, type=float, required=False, help="Sparsity level (between 0 and 1)")

parser.add_argument('--model_path', default="./vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx", type=str, required=False, help="Model path")

parser.add_argument('--sub_folder', default="train_hard", type=str, required=False, help="Path to subfolder")

parser.add_argument('--digit', default=0, type=int, required=False, help="Digit")

args = parser.parse_args()

SEED = 70
SPARSITY = args.sparsity
FOLDER_PATH = os.path.dirname(__file__)
# MODEL_PATH=f"{FOLDER_PATH}/benchmarks/test/test_nano.onnx"
MODEL_PATH=args.model_path

SUBFOLDER=args.sub_folder

MODEL_NAME, _ = os.path.splitext(os.path.basename(MODEL_PATH))
if SPARSITY == 0:
    SAVE_MODEL_PATH = f"{FOLDER_PATH}/pytorch_model/{SUBFOLDER}_0.5"
else:
    SAVE_MODEL_PATH = f"{FOLDER_PATH}/pytorch_model/{SUBFOLDER}_{SPARSITY}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DENSE_PATH = f"{SAVE_MODEL_PATH}/{MODEL_NAME}.pth"
SPARSE_PATH = f"{SAVE_MODEL_PATH}/{MODEL_NAME}_{SPARSITY}.pth"

digit = args.digit

def set_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def count_neurons(nn_model, model_path):

    print("Device to train: ", DEVICE)

    train_loader = MNISTDataset(train=True, batch_size=64).get_data()
    test_loader = MNISTDataset(train=False, batch_size=64).get_data()

    nn_model.load_state_dict(torch.load(model_path))

    nn_model = nn_model.to(device=DEVICE)

    trainer = ModelTrainer(nn_model, max_epochs=1, learning_rate=1, device=DEVICE)

    test_score = trainer.calculate_score(test_loader)

    model, neuron_freq, weight_freq, abs_weights = trainer.count_neurons(train_loader, digit=digit)

    return model, neuron_freq, weight_freq, abs_weights, test_score

def dense_model_train():

    # Convert to PyTorch
    onnx_model = convert(onnx.load(MODEL_PATH))

    torch_model = ONNXModel(onnx_model)

    return torch_model

set_seed(SEED)

if SPARSITY == 0:
    model, neuron, weight, abs_weights, acc = count_neurons(dense_model_train(), DENSE_PATH)

else:
    model, neuron, weight, abs_weights, acc = count_neurons(dense_model_train(), SPARSE_PATH)

save_path = f'./neuron/{SUBFOLDER}_{MODEL_NAME}_{SPARSITY}_rundigit_{digit}.npz'

print("Saving: ", save_path)

np.savez(save_path, len(neuron), len(weight), acc, *neuron, *weight, *abs_weights)

#dense_freq, sparse_freq = np.concatenate(dense_freq), np.concatenate(sparse_freq)

#plt.hist([dense_freq, sparse_freq], edgecolor='black', alpha=0.4)

#plt.hist(np.concatenate(sparse_fre, bins=range(0, 64, 2), edgecolor='black', alpha=0.4)


#plt.savefig("neuron.png")

print("Num parameters : ", model.count_parameters())


