from Dataset import MNISTDataset
import argparse
import numpy as np
import torch
from copy import deepcopy
import random
from Model import SimpleNN
from Pruner import ModelTrainer, Pruner
import os

parser = argparse.ArgumentParser(description="Model state dictionary checker and saver.")
parser.add_argument('--seed', type=int, required=True, help="Random seed for reproducibility")
parser.add_argument('--sparsity', type=float, required=True, help="Sparsity level (between 0 and 1)")
args = parser.parse_args()

seed = args.seed
sparsity = args.sparsity
max_epochs = 1
learning_rate = 0.01
train_first = False
n_rounds = 1
file_path = f"./results/seed_{seed}/model"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device to train: ", device)

def set_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def train_model(n_rounds, sparsity, train_first):

    print("Device to train: ", device)

    x_train, y_train = MNISTDataset(train=True).get_data()
    x_test, y_test = MNISTDataset(train=False).get_data()

    nn_model = SimpleNN()

    if train_first:
        nn_model.load_state_dict(torch.load(f'{file_path}/dense.pth'))

    nn_model = nn_model.to(device=device)
    trainer = ModelTrainer(nn_model, max_epochs=max_epochs, learning_rate= learning_rate, device=device)

    initial_weights = deepcopy(nn_model.state_dict())
    total_parameters = nn_model.count_parameters()
    prune_pc_per_round = 1 - (1 - sparsity) ** (1 / n_rounds)

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
    trainer.train(x_train, y_train)
    nn_model.apply_mask()
    print("Non zero params: ", nn_model.count_parameters())



    validation_score = trainer.calculate_score(x_test, y_test)
    print(f"Validation set score: {validation_score:.4}")

    # print(nn_model.get_weights())

    nn_model = nn_model.to("cpu")
    return nn_model, validation_score

def dense_model_train():

    print("Dense Model")

    x_train, y_train = MNISTDataset(train=True).get_data()
    x_test, y_test = MNISTDataset(train=False).get_data()

    nn_model = SimpleNN()
    nn_model = nn_model.to(device=device)
    trainer = ModelTrainer(nn_model, max_epochs=max_epochs, learning_rate= learning_rate, device=device)

    trainer.train(x_train, y_train)

    print("Non zero params: ", nn_model.count_parameters())

    validation_score = trainer.calculate_score(x_test, y_test)

    nn_model = nn_model.to("cpu")
    return nn_model, validation_score

set_seed(seed)

dense_model, dense_validation = dense_model_train()
dense_path = f"{file_path}/dense.pth"

if os.path.exists(dense_path):
    print(f"The file {dense_path} exists.")
    
    # Load the state dictionary from the file
    loaded_state_dict = torch.load(dense_path)
    
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
        print(f"The file {dense_path} does not exist. Saving the model state dictionary.")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(dense_path), exist_ok=True)
        
        torch.save(dense_model.state_dict(), f'{dense_path}')



sparse_model, sparse_validation = train_model(n_rounds=n_rounds, sparsity=sparsity, train_first=train_first)

torch.save(sparse_model.state_dict(), f'{file_path}/sparse_{int(sparsity * 100)}_{train_first}.pth')

print(dense_model.count_parameters())
print(sparse_model.count_parameters())


