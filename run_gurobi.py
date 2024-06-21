import torch
from Model import SimpleNN
from Dataset import MNISTDataset
import numpy as np
import random
import csv
import os
from gurobi_MNIST import solve_with_gurobi
import argparse

parser = argparse.ArgumentParser(description="Model state dictionary checker and saver.")
parser.add_argument('--seed', type=int, required=True, help="Random seed for reproducibility")
parser.add_argument('--sparsity', type=float, required=True, help="Sparsity level (between 0 and 1)")
parser.add_argument('--callback', type=str, default="none", choices=["none", "dense_passing"], required=False, help="Choose Callback function")


args = parser.parse_args()


SEED = args.seed
SPARSITY = args.sparsity
TRAIN_FIRST = False
DELTA = 15
TIME_LIMIT = 3600
FOLDER_PATH = f"./results/seed_{SEED}"
CALLBACK = args.callback
NUM_IMAGES_PER_DIGIT=2


def append_to_csv(file_name, data_dict):
    # Check if the file already exists
    file_exists = os.path.isfile(file_name)

    # Open the file in append mode
    with open(file_name, 'a', newline='') as csvfile:
        fieldnames = data_dict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If the file does not exist, write the header
        if not file_exists:
            writer.writeheader()

        # Write the data dictionary as a new row
        writer.writerow(data_dict)


def run_gurobi(model, dense_model, image_x):

    nn_regression = torch.nn.Sequential(*model.layers[:-1])

    ex_prob = nn_regression.forward(image_x)

    top2_value, top2_index = torch.topk(ex_prob, 2)

    correct_label = top2_index[0].item()
    wrong_label = top2_index[1].item()

    image = image_x.numpy()

    print('!Gurobi Start!')
    x_max, max_, time_count = solve_with_gurobi(nn_regression, dense_model, image, DELTA, ex_prob, wrong_label, correct_label, TIME_LIMIT, CALLBACK)

    # print(x_max, max_, time_count)
    return x_max, max_, time_count, correct_label, wrong_label

def import_model(file_name):

    model = SimpleNN()

    model.load_state_dict(torch.load(file_name))

    return model

def set_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

print("Seed: ", SEED)
set_seed(SEED)

dense_model = import_model(f"{FOLDER_PATH}/model/dense.pth")

sparse_model = import_model(f"{FOLDER_PATH}/model/sparse_{int(SPARSITY * 100)}_{TRAIN_FIRST}.pth")
sparse_model = import_model(f"{FOLDER_PATH}/model/dense.pth") if SPARSITY == 0 else sparse_model

print(dense_model.count_parameters())
print(sparse_model.count_parameters())

train_data = MNISTDataset(train=True)
x_train, y_train = MNISTDataset(train=True).get_data()

sort_data = []

data_dict = {}
data_dict["seed"] = SEED
data_dict["sparsity"] = SPARSITY
data_dict["train_first"] = TRAIN_FIRST
data_dict["delta"] = DELTA
data_dict["time_limit"] = TIME_LIMIT

for i in range(10):
    sort_data.append([])

for i in range(len(x_train)):
    sort_data[y_train[i]].append((x_train[i], i))

for i in range(10):

    random_selection = random.sample(sort_data[i], NUM_IMAGES_PER_DIGIT)

    for j in random_selection:

        print("Running: ", i)

        x_max, _max, time_count, correct_label, wrong_label = run_gurobi(sparse_model, dense_model, j[0])

        data_dict["image_index"] = j[1]
        data_dict["label"] = i
        data_dict["predict_label"] = correct_label
        data_dict["wrong_label"] = wrong_label
        data_dict["x_max"] = x_max
        data_dict["max"] = _max
        data_dict["runtime"] = time_count
        data_dict["method"] = CALLBACK


        if x_max != None:

            sparse_model.eval()
            dense_model.eval()
            with torch.no_grad():
                dense_predicted = torch.argmax(dense_model.forward(torch.tensor(x_max)))
                sparse_predicted = torch.argmax(sparse_model.forward(torch.tensor(x_max)))
                origin_dense = torch.argmax(dense_model.forward(j[0]))
                origin_sparse = torch.argmax(sparse_model.forward(j[0]))

            data_dict["valid_image"] = (dense_predicted != origin_dense).item()


            save_image_path = f"{FOLDER_PATH}/images"
            os.makedirs(save_image_path, exist_ok=True)

            # View image
            # train_data.save_image(j[1], origin_dense, origin_sparse, x_max, dense_predicted, sparse_predicted, f"{save_image_path}/{j[1]}_{int(SPARSITY * 100)}_{TRAIN_FIRST}.png")

        else:
            data_dict["valid_image"] = False


        append_to_csv(f"{FOLDER_PATH}/summary.csv", data_dict=data_dict)





