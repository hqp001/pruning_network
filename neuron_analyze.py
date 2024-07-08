from Model import ONNXModel
from onnx2torch import convert
import onnx
import numpy as np
import gzip
import torch

def get_model(filename, torchfile):
    # Convert to PyTorch
    onnx_model = convert(onnx.load(filename))

    torch_model = ONNXModel(onnx_model)

    torch_model.load_state_dict(torch.load(torchfile))

    return torch_model

def get_input(ce_path):

    if ce_path.endswith('.gz'):
        with gzip.open(ce_path, 'rb') as f:
            content = f.read().decode('utf-8')
    else:
        with open(ce_path, 'r', encoding='utf-8') as f:
            content = f.read()

    content = content.replace('\n', ' ').strip()
    assert content[0] == '(' and content[-1] == ')'
    content = content[1:-1]

    x_list = []
    y_list = []

    parts = content.split(')')
    for part in parts:
        part = part.strip()

        if not part:
            continue

        assert part[0] == '('
        part = part[1:]

        name, num = part.split(' ')
        assert name[0:2] in ['X_', 'Y_']

        if name[0:2] == 'X_':
            assert int(name[2:]) == len(x_list)
            x_list.append(float(num))
        else:
            assert int(name[2:]) == len(y_list)
            y_list.append(float(num))


    return x_list, y_list

def passing(model, x):

    activations = {}
    ret_list = []
    def hook_fn(module, input, output):
        #print(module)
        #print(input.shape)
        activations[module] = output.detach().numpy()

    for layer in model.model.children():
        layer.register_forward_hook(hook_fn)

#    print(x.shape)

    output = model(x)

    last_activation = [np.ones((len(x), 784), dtype=int)]

    for layer, activation in activations.items():
        output_activation = activation
#        print(layer)
        if isinstance(layer, torch.nn.ReLU):
            output_activation = np.where(output_activation > 0, 1, 0).astype(int)
            indexes_list = np.where(output_activation == 1)[0].tolist()
#            print(output_activation)
            #print(indexes_list)
            ret_list.append(indexes_list)

        if isinstance(layer, torch.nn.Linear):
#            print(output_activation)
            pass

    return ret_list

def load_data(file_path):

    loaded_data = np.load(file_path)
    n_layers = loaded_data['arr_0']
    n_weights = loaded_data['arr_1']
    acc = loaded_data['arr_2']
    neurons = []
    weights = []
    abs_weights = []

    for i in range(n_layers):
        neurons.append(loaded_data[f'arr_{i + 3}'])

    for i in range(n_weights):
        weights.append(loaded_data[f'arr_{i + 3 + n_layers}'])

    for i in range(n_weights):
        abs_weights.append(loaded_data[f'arr_{i + 3 + n_layers + n_weights}'])

    return acc, neurons, weights, abs_weights


model_name = "mnist-net_256x6"
subfolder=["alpha_beta_crown", "train_hard_0.8_dense_passing", "marabou", "verinet", "dense_0"]

output = get_input(f"./vnncomp2022_results/{subfolder[3]}/mnist_fc/{model_name}_prop_6_0.05.counterexample.gz")

model = get_model(f"./vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/{model_name}.onnx", f"./pytorch_model/dense_0/{model_name}.pth")
#print(output)
data = load_data(f"./neuron/sgd_trained_{model_name}_0.0.npz")
neuron_freq = data[1]


x = torch.tensor(output[0], dtype=torch.float32)

y = model.forward(x)
print(output[1])
print(y)

print(neuron_freq)
indexes = passing(model, x)
print(indexes)
for idx, layer in enumerate(indexes):
    for i in layer:
        print(f'{i}-{neuron_freq[idx][i]}  ', end="")
    print()
