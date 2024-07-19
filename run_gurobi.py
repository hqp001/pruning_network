import os
import random
import argparse

import torch
import numpy as np
import onnx
from onnx2torch import convert
from vnnlib.compat import read_vnnlib_simple

from utils.Model import ONNXModel, DoubleModel, count_params
from Solver.GurobiSolver import solve_with_gurobi

parser = argparse.ArgumentParser(description="Model state dictionary checker and saver.")
parser.add_argument('--sparsity',
        default=0.9,
        type=float,
        required=False,
        help="Sparsity level (between 0 and 1)")
parser.add_argument('--model_path',
        type=str,
        required=False,
        default=f"./vnncomp2022_benchmarks/benchmarks/oval21/onnx/cifar_base_kw.onnx",
        help='Path to the model file')
parser.add_argument('--instance_path',
        type=str,
        required=False,
        default=f"./vnncomp2022_benchmarks/benchmarks/oval21/vnnlib/cifar_base_kw-img3568-eps0.030457516339869282.vnnlib",
        help='Path to the instance file')
parser.add_argument('--time_limit',
        type=int,
        required=False,
        default=300,
        help='Time limit for the program in seconds (default is 3600 seconds)')
parser.add_argument('--output_path', type=str, required=False, default="output.txt", help="Output path")
parser.add_argument('--subfolder', type=str, required=False, default="a", help="Subfolder name")
parser.add_argument('--callback', type=str, required=False, default="none", help="Name of call back function")
parser.add_argument('--double', action="store_true", help="Convert to model twice as large")

args = parser.parse_args()

SPARSITY = args.sparsity
TIME_LIMIT = args.time_limit

FOLDER_PATH = os.path.dirname(__file__)

ONNX_PATH = f"{args.model_path}"
INSTANCE_PATH = f"{args.instance_path}"
OUTPUT_PATH = f"{args.output_path}"

SUBFOLDER = args.subfolder

MODEL_NAME, _ = os.path.splitext(os.path.basename(ONNX_PATH))
LOAD_MODEL_PATH = f"{FOLDER_PATH}/pytorch_model/{SUBFOLDER}_{SPARSITY}"
DENSE_PATH = f"{LOAD_MODEL_PATH}/{MODEL_NAME}.pth"
SPARSE_PATH = f"{LOAD_MODEL_PATH}/{MODEL_NAME}_{SPARSITY}.onnx"

DOUBLE = args.double

CALLBACK = args.callback

CATEGORY = os.path.basename(os.path.dirname(os.path.dirname(ONNX_PATH)))

if CATEGORY == "oval21":

    INPUT_SIZE = 3072
    OUTPUT_SIZE = 10
    INPUT_SHAPE = (1, 3, 32, 32)

elif CATEGORY == "mnist_fc":

    INPUT_SIZE = 784
    OUTPUT_SIZE = 10
    INPUT_SHAPE = (1, 28, 28)

else:

    raise ValueError(f"Unknown Category: {CATEGORY}")

def run_gurobi(sparse_model, dense_model, input):

    input_range = input[0]
    output_range = input[1]

    lb_image = np.array([i[0] for i in input_range]).reshape(*INPUT_SHAPE)
    ub_image = np.array([i[1] for i in input_range]).reshape(*INPUT_SHAPE)
    correct_label = np.argmax(output_range[0][0][0])

    random_image = torch.tensor((lb_image + ub_image) / 2, dtype=torch.float32)
    ex_prob = dense_model.forward(random_image)
    top2_value, top2_index = torch.topk(ex_prob, 2)
    wrong_label = top2_index[0][1].item()

    x_max, max_, time_count = solve_with_gurobi(sparse_model, dense_model, [lb_image, ub_image], correct_label, wrong_label, TIME_LIMIT, CALLBACK)

    # print(x_max, max_, time_count)
    #print(max_, time_count, correct_label, wrong_label)
    return x_max, max_, time_count, correct_label, wrong_label

def import_model(file_name):

    onnx_model = convert(onnx.load(file_name))

    return onnx_model

def get_image(file_name):

    result = read_vnnlib_simple(file_name, INPUT_SIZE, OUTPUT_SIZE)

    return result[0][0], result[0][1]


def set_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def print_result_to_file(result, x_max, y_max):
    with open(OUTPUT_PATH, "w") as f:
        f.write(result + "\n")
        if x_max is not None:
            f.write("(")
            for idx, value in enumerate(x_max):
                f.write(f"(X_{idx} {value})\n")
            for idx, value in enumerate(y_max):
                f.write(f"(Y_{idx} {value})\n")
            f.write(")")


dense_model = import_model(ONNX_PATH)

if SPARSITY == 0:
    sparse_model = import_model(ONNX_PATH)
else:
    sparse_model = import_model(SPARSE_PATH)

print("Dense # Params: ", count_params(dense_model))
print("Sparse # Params: ", count_params(sparse_model))

x_max, _max, time_count, correct_label, wrong_label = run_gurobi(sparse_model, dense_model, get_image(INSTANCE_PATH))

if x_max is not None:
    dense_model.eval()
    with torch.no_grad():
        y_max = dense_model.forward(torch.tensor(x_max, dtype=torch.float32))
        dense_output = torch.argmax(y_max).item()

else:
    dense_output = None

if dense_output != None and dense_output != correct_label:
    y_max = dense_model.forward(torch.tensor(x_max, dtype=torch.float32)).detach().numpy()
    print_result_to_file("sat", x_max.reshape(-1), y_max.reshape(-1))

else:
    print_result_to_file("unsat", None, None)

#append_to_csv(f"{file_path}/summary_{sparsity}.csv", data_dict=data_dict)


