import torch
from Model import SimpleNN
import gurobipy as gp

def add_predictor_constr(gurobi_model, nn_model, x):

    sequential = []

    sequential.append(x)

    for idx, layer in enumerate(nn_model.layers):
        if (isinstance(layer, torch.nn.Linear)):
            linear = add_linear_constr(gurobi_model, sequential[-1], weight = layer.weight.data.numpy(), bias = layer.bias.data.numpy(), name = f"layer_{idx}")

            sequential.append(linear)

        elif (isinstance(layer, torch.nn.ReLU)):
            relu = add_relu_constr(gurobi_model, sequential[-1], name = f"relu_{idx}")

        elif (isinstance(layer, torch.nn.Softmax)):
            pass

        else:
            raise ValueError("Unknown Layer")

    y = sequential[-1]

    return y

def add_linear_constr(gurobi_model, input_layer, weight, bias, name="linear"):

    input_size = weight.shape[1]
    output_size = weight.shape[0]

    assert(input_size == input_layer.size and output_size == bias.size)

    print(weight.shape, input_layer.shape, bias.shape)

    output_layer = ( weight @ input_layer ) + bias

    return output_layer

def add_relu_constr(gurobi_model, input_layer, name):

    input_size = input_layer.size

    neuron_layer = gurobi_model.addMVar(input_size, vtype=gp.GRB.BINARY, name = f"neuron_{name}")

    output_layer = gurobi_model.addMVar(input_size, name=name)

    for j in range(input_size):
        gurobi_model.addConstr( (neuron_layer[j] == 0) >> (input_layer[j] <= 0) )
        gurobi_model.addConstr( (neuron_layer[j] == 0) >> (output_layer[j] == 0) )
        gurobi_model.addConstr( (neuron_layer[j] == 1) >> (output_layer[j] == input_layer[j]) )
        gurobi_model.addConstr( output_layer[j] >= 0 )

    return output_layer


def test():

    torch.manual_seed(70)

    nn_model = SimpleNN()

    gurobi_model = gp.Model()

    x = gurobi_model.addMVar(4, name="x")

    add_predictor_constr(gurobi_model, nn_model, x)

    gurobi_model.update()

    print(gurobi_model.getVars())

    gurobi_model.write("model.lp")

