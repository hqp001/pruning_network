import torch
from Model import SimpleNN
import gurobipy as gp

def add_predictor_constr(gurobi_model, nn_model, x, y):

    sequential = []

    neuron_constr_list = []

    sequential.append(x)

    for idx, layer in enumerate(nn_model.layers):
        if (isinstance(layer, torch.nn.Linear)):
            linear = add_linear_constr(gurobi_model, sequential[-1], weight = layer.weight.data.numpy(), bias = layer.bias.data.numpy(), name = f"linear_{idx}")

            sequential.append(linear)


        elif (isinstance(layer, torch.nn.ReLU)):
            relu, neuron_constr = add_relu_constr(gurobi_model, sequential[-1], name = f"relu_{idx}")

            sequential.append(relu)

            print(layer, idx)
            neuron_var = gurobi_model.addVar(vtype=gp.GRB.BINARY, name="Neuron constr")
            gurobi_model.addConstr((neuron_var == 1) >> (neuron_constr <= 5))
            neuron_constr_list.append(neuron_var)

        elif (isinstance(layer, torch.nn.Softmax)):
            pass

        else:
            raise ValueError("Unknown Layer")

    gurobi_model.addConstr(gp.quicksum(neuron_constr_list) >= 1)

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


def test():

    torch.manual_seed(70)

    nn_model = SimpleNN()

    gurobi_model = gp.Model()

    x = gurobi_model.addMVar(4, name="x")

    add_predictor_constr(gurobi_model, nn_model, x)

    gurobi_model.update()

    print(gurobi_model.getVars())

    gurobi_model.write("model.lp")

