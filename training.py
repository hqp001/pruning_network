from Dataset import MNISTDataset
import argparse
import numpy as np
import torch
from copy import deepcopy
import random
from Model import ONNXModel
from Pruner import ModelTrainer, Pruner
from onnx2torch import convert
import onnx
import os

parser = argparse.ArgumentParser(description="Model state dictionary checker and saver.")
parser.add_argument('--sparsity', default=0.5, type=float, required=False, help="Sparsity level (between 0 and 1)")
parser.add_argument('--model_path', default="mnist_fc/onnx/mnist-net_256x2.onnx", type=str, required=False, help="Model path")


args = parser.parse_args()

SEED = 70
SPARSITY = args.sparsity
FOLDER_PATH = os.path.dirname(__file__)
# MODEL_PATH=f"{FOLDER_PATH}/benchmarks/test/test_nano.onnx"
MODEL_PATH=f"{FOLDER_PATH}/benchmarks/{args.model_path}"


MODEL_NAME, _ = os.path.splitext(os.path.basename(MODEL_PATH))
SAVE_MODEL_PATH = f"{FOLDER_PATH}/pytorch_model/sparse_no_train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DENSE_PATH = f"{SAVE_MODEL_PATH}/{MODEL_NAME}.pth"
SPARSE_PATH = f"{SAVE_MODEL_PATH}/{MODEL_NAME}_{SPARSITY}.pth"

TEST = True if 'test' in MODEL_PATH else False

max_epochs = 70
learning_rate = 0.001
train_first = False
n_rounds = 10

def set_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def train_model(nn_model, n_rounds):

    print("Device to train: ", DEVICE)

    x_train, y_train = MNISTDataset(train=True).get_data()
    x_test, y_test = MNISTDataset(train=False).get_data()

    if not train_first:
        print("Not using trained network")
        nn_model.apply(initialize_weights)

    nn_model = nn_model.to(device=DEVICE)

    trainer = ModelTrainer(nn_model, max_epochs=max_epochs, learning_rate= learning_rate, device=DEVICE)

    initial_weights = deepcopy(nn_model.state_dict())
    total_parameters = nn_model.count_parameters()
    prune_pc_per_round = 1 - (1 - SPARSITY) ** (1 / n_rounds)

    # print(nn_model.get_weights())

    print("Total Params:", total_parameters)

    for round in range(n_rounds):
        print(f"\nPruning round {round} of {n_rounds}")

        # Fit the model to the training data
        trainer.train(X=x_train, y=y_train)

        pruned_model = Pruner(model=nn_model, trainer=trainer, sparsity=prune_pc_per_round).prune().get_model()

        # Reset model
        nn_model.initialize_weights(initial_weights)

        # print(f"Model accuracy: {accuracy:.3f}%")
        # print(f"New parameters: {n_pruned_parameters}/{total_parameters}")

    # Train final model
    trainer.train(X=x_train, y=y_train)
    nn_model.apply_mask()

    validation_score = trainer.calculate_score(x_test, y_test)

    print(validation_score)

    # print(nn_model.get_weights())

    nn_model = nn_model.to("cpu")
    return nn_model, validation_score

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

dense_model = dense_model_train()

if os.path.exists(DENSE_PATH):
    print(f"The file {DENSE_PATH} exists.")

    # Load the state dictionary from the file
    loaded_state_dict = torch.load(DENSE_PATH)

    # Get the current model's state dictionary
    current_state_dict = dense_model.state_dict()

    # Compare the state dictionaries
    state_dicts_equal = all(torch.equal(current_state_dict[key], loaded_state_dict[key]) for key in current_state_dict)

    if state_dicts_equal:
        print("The model state dictionary is the same as in the file.")
    else:
        print("The model state dictionary is different from the one in the file.")

else:
        # Save the model state dictionary
        print(f"The file {DENSE_PATH} does not exist. Saving the model state dictionary.")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(DENSE_PATH), exist_ok=True)

        torch.save(dense_model.state_dict(), f'{DENSE_PATH}')

sparse_model = None

if not TEST:

    sparse_model, validation_score = train_model(dense_model_train(), n_rounds=n_rounds)

    torch.save(sparse_model.state_dict(), SPARSE_PATH)

    print(validation_score)

else:

    sparse_model = dense_model_train()


print(dense_model.count_parameters())
print(sparse_model.count_parameters())

