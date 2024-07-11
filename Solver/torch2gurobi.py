import torch
from Model import SimpleNN
import gurobipy as gp
import numpy as np

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

    return neurons


#data = load_data(f"./neuron/sgd_trained_mnist-net_256x{model_name}_0.0.npz")

def add_predictor_constr(gurobi_model, nn_model, x, y):

    sequential = []

    #neuron_constr_list = []

    sequential.append(x)

    n_relu = 0
    for idx, layer in enumerate(nn_model.layers):
        n_relu += isinstance(layer, torch.nn.ReLU)

    #neuron_data = load_data(f"./neuron/sgd_trained_mnist-net_256x{n_relu}_0.0.npz")

    for idx, layer in enumerate(nn_model.layers):
        if (isinstance(layer, torch.nn.Linear)):
            linear = add_linear_constr(gurobi_model, sequential[-1], weight = layer.weight.data.numpy(), bias = layer.bias.data.numpy(), name = f"linear_{idx}")

            sequential.append(linear)

        elif (isinstance(layer, torch.nn.ReLU)):
            relu, neuron_constr = add_relu_constr(gurobi_model, sequential[-1], name = f"relu_{idx}")

            sequential.append(relu)

            #print(layer, idx)
            #neuron_var = gurobi_model.addVar(vtype=gp.GRB.BINARY, name="neuron_constr")
            #gurobi_model.addConstr((neuron_var == 1) >> (neuron_constr <= 2))
            #neuron_constr_list.append(neuron_var)

        elif (isinstance(layer, torch.nn.Conv2d)):

            pass

        elif (isinstance(layer, torch.nn.Softmax)):
            pass

        else:
            raise ValueError("Unknown Layer")

    #gurobi_model.addConstr(gp.quicksum(neuron_constr_list) >= 1)

    output = sequential[-1]

    assert(y.size == output.size)

    for i in range(y.size):

        gurobi_model.addConstr(output[i] == y[i])

    return y

def add_linear_constr(gurobi_model, input_layer, weight, bias, name="linear"):

    input_size = weight.shape[1]
    output_size = weight.shape[0]

    #print(weight)

    assert(input_size == input_layer.size and output_size == bias.size)

    output_layer = ( input_layer @ weight.T ) + bias

    return output_layer

def add_linear_constr2(gurobi_model, input_layer, weight, bias, name="linear"):

    input_size = weight.shape[1]
    output_size = weight.shape[0]

    assert(input_size == input_layer.size and output_size == bias.size)

    output_layer = gurobi_model.addMVar(output_size, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name=name)

    gurobi_model.addConstr(output_layer == ( input_layer @ weight.T ) + bias)

    return output_layer

def add_relu_constr2(gurobi_model, input_layer, name):

    input_size = input_layer.size
    output_layer = gurobi_model.addMVar(input_size, name=name)
    for j in range(input_size):
       gurobi_model.addConstr(output_layer[j] == gp.max_([input_layer[j]], constant=0.0))

    return output_layer

def add_relu_constr(gurobi_model, input_layer, name):

    input_size = input_layer.size

    neuron_layer = gurobi_model.addMVar(input_size, vtype=gp.GRB.BINARY, name = f"neuron_{name}")

    output_layer = gurobi_model.addMVar(input_size, name=name)
    neuron_sum = 0
    for j in range(input_size):
        gurobi_model.addConstr( (neuron_layer[j] == 0) >> (input_layer[j] <= 0) )
        gurobi_model.addConstr( (neuron_layer[j] == 0) >> (output_layer[j] == 0) )
        gurobi_model.addConstr( (neuron_layer[j] == 1) >> (output_layer[j] == input_layer[j]) )
        gurobi_model.addConstr( output_layer[j] >= 0 )
        neuron_sum += neuron_layer[j]


    return output_layer, neuron_sum

def score(a):

    return 100 - int(abs(a - 0.5) * 100) - 50

def add_relu_constr3(gurobi_model, input_layer, neuron_freq, name):

    input_size = input_layer.size

    neuron_layer = gurobi_model.addMVar(input_size, vtype=gp.GRB.BINARY, name = f"neuron_{name}")

    output_layer = gurobi_model.addMVar(input_size, name=name)
    neuron_sum = 0
    for j in range(input_size):
        print(j, neuron_freq[j], score(neuron_freq[j]))
        gurobi_model.addConstr( (neuron_layer[j] == 0) >> (input_layer[j] <= 0) )
        gurobi_model.addConstr( (neuron_layer[j] == 0) >> (output_layer[j] == 0) )
        gurobi_model.addConstr( (neuron_layer[j] == 1) >> (output_layer[j] == input_layer[j]) )
        gurobi_model.addConstr( output_layer[j] >= 0 )
        neuron_layer[j].BranchPriority = score(neuron_freq[j])
        neuron_sum += neuron_layer[j]


    return output_layer, neuron_sum

def test():

    torch.manual_seed(70)

    nn_model = SimpleNN()

    gurobi_model = gp.Model()

    x = gurobi_model.addMVar(4, name="x")
    y = gurobi_model.addMVar(10, name="y")

    add_predictor_constr(gurobi_model, nn_model, x, y)

    gurobi_model.update()

    print(gurobi_model.getVars())

    gurobi_model.write("model.lp")

