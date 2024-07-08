import torch
import pandas as pd
from Model import ONNXModel
from onnx2torch import convert
import onnx
import numpy as np
import random
import os
from gurobi_MNIST import solve_with_gurobi_and_record
import argparse
from vnnlib.compat import read_vnnlib_simple

parser = argparse.ArgumentParser(description="Model state dictionary checker and saver.")
parser.add_argument('--sparsity',
        default=0.5,
        type=float,
        required=False,
        help="Sparsity level (between 0 and 1)")
parser.add_argument('--model_path',
        type=str,
        required=False,
        default=f"./vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx",
        help='Path to the model file')
parser.add_argument('--instance_path',
        type=str,
        required=False,
        default=f"./vnncomp2022_benchmarks/benchmarks/mnist_fc/vnnlib/prop_12_0.05.vnnlib",
        help='Path to the instance file')
parser.add_argument('--time_limit',
        type=int,
        required=False,
        default=300,
        help='Time limit for the program in seconds (default is 3600 seconds)')
parser.add_argument('--output_path',
        type=str,
        required=False,
        default="output.txt",
        help="Output path")
parser.add_argument('--subfolder',
        type=str,
        required=False,
        default="sparse_std",
        help="Subfolder name")
parser.add_argument('--callback',
        type=str,
        required=False,
        default="none",
        help="Name of call back function")


args = parser.parse_args()

SPARSITY = args.sparsity
TIME_LIMIT = args.time_limit

FOLDER_PATH = os.path.dirname(__file__)

ONNX_PATH = f"{args.model_path}"
INSTANCE_PATH = f"{args.instance_path}"
OUTPUT_PATH = f"{args.output_path}"


TEST = True if 'test' in ONNX_PATH else False
SUBFOLDER = "test" if TEST else args.subfolder

MODEL_NAME, _ = os.path.splitext(os.path.basename(ONNX_PATH))
LOAD_MODEL_PATH = f"{FOLDER_PATH}/pytorch_model/{SUBFOLDER}"
DENSE_PATH = f"{LOAD_MODEL_PATH}/{MODEL_NAME}.pth"
SPARSE_PATH = f"{LOAD_MODEL_PATH}/{MODEL_NAME}_{SPARSITY}.pth"

CALLBACK = args.callback

def run_gurobi(model, dense_model, input):

    image_range = input[0]

    output_range = input[1]

    first_image = []

    for i in image_range:

        first_image.append((image_range[0][1] + image_range[0][0]) / 2)

    #print(first_image)

    ex_prob = model.forward(torch.tensor(first_image))

    top2_value, top2_index = torch.topk(ex_prob, 2)

    correct_label = np.argmax(output_range[0][0][0])

    #print(correct_label)

    wrong_label = None

    if top2_index[0].item() != correct_label:
        wrong_label = top2_index[0].item()

    else:
        wrong_label = top2_index[1].item()

    x_max, max_, time_count = solve_with_gurobi_and_record(model, dense_model, image_range, output_range, TIME_LIMIT, correct_label, wrong_label, CALLBACK)

    # print(x_max, max_, time_count)
    #print(max_, time_count, correct_label, wrong_label)
    return x_max, max_, time_count, correct_label, wrong_label

def import_model(file_name):

    onnx_model = convert(onnx.load(ONNX_PATH))

    torch_model = ONNXModel(onnx_model)

    torch_model.load_state_dict(torch.load(file_name))

    return torch_model

def get_image(file_name):

    result = read_vnnlib_simple(file_name, 784, 10)

    return result[0][0], result[0][1]


def set_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def print_result_to_file(result, x_max, y_max):
    with open(OUTPUT_PATH, "w") as f:
        f.write(result + "\n")
        if x_max != None:
            f.write("(")
            for idx, value in enumerate(x_max):
                f.write(f"(X_{idx} {value})\n")
            for idx, value in enumerate(y_max):
                f.write(f"(Y_{idx} {value})\n")
            f.write(")")


SPARSE_PATH = DENSE_PATH if SPARSITY == 0 else SPARSE_PATH

dense_model = import_model(DENSE_PATH)
sparse_model = import_model(SPARSE_PATH)

print(dense_model.count_parameters())
print(sparse_model.count_parameters())

x_max, _max, time_count, correct_label, wrong_label = run_gurobi(sparse_model, dense_model, get_image(INSTANCE_PATH))

if x_max != None:

    dense_model.eval()
    sparse_model.eval()
    with torch.no_grad():
        dense_output = torch.argmax(dense_model.forward(torch.tensor(x_max))).item()
else:
    dense_output = None

if dense_output != None and dense_output != correct_label:
    print_result_to_file("sat", x_max, dense_model.forward(torch.tensor(x_max)).tolist())
else:
    print_result_to_file("unsat", None, None)

#append_to_csv(f"{file_path}/summary_{sparsity}.csv", data_dict=data_dict)


