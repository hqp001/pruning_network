import torch
import numpy as np
import gurobipy as gp

def add_predictor_constr(gurobi_model, nn_model, x, y):

    sequential = []

    sequential.append(x)

    for idx, layer in enumerate(nn_model.layers):
        if (isinstance(layer, torch.nn.Linear)):
            linear = add_linear_constr(gurobi_model, sequential[-1], weight = layer.weight.data.numpy(), bias = layer.bias.data.numpy(), name = f"linear_{idx}")

            sequential.append(linear)

        elif (isinstance(layer, torch.nn.ReLU)):
            relu = add_relu_constr(gurobi_model, sequential[-1], name = f"relu_{idx}")

            sequential.append(relu)

        elif (isinstance(layer, torch.nn.Conv2d)):
            conv = add_conv2d_constr(gurobi_model, sequential[-1], name = f"conv2d_{idx}")

            sequential.append(conv)

            print(aa)

        elif (isinstance(layer, torch.nn.Softmax)):
            # Skipping Softmax layer
            pass

        elif (isinstance(layer, torch.nn.Flatten)):

            flatten = add_flatten_constr(gurobi_model, sequential[-1], layer.start_dim, layer.end_dim, f"flattend_{idx}")

            sequential.append(flatten)

        else:
            raise ValueError(f"Unknown Layer: {type(layer)}")

    output = sequential[-1]

    assert(y.size == output.size)

    for i in range(y.shape[1]):

        gurobi_model.addConstr(output[0][i] == y[0][i])

    return y

def add_conv2d_constr(gurobi_model, input_layer, name = "conv2d"):

    gurobi_model.update()
    print(input_layer)





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

    input_size = input_layer.shape

    neuron_layer = gurobi_model.addMVar(input_size, vtype=gp.GRB.BINARY, name = f"neuron_{name}")

    gurobi_model._binary.append(neuron_layer)

    output_layer = gurobi_model.addMVar(input_size, name=name)
    for j in range(input_size[1]):
        gurobi_model.addConstr( (neuron_layer[0][j] == 0) >> (input_layer[0][j] <= 0) )
        gurobi_model.addConstr( (neuron_layer[0][j] == 0) >> (output_layer[0][j] == 0) )
        gurobi_model.addConstr( (neuron_layer[0][j] == 1) >> (output_layer[0][j] == input_layer[0][j]) )
        gurobi_model.addConstr( output_layer[0][j] >= 0 )


    return output_layer

def add_flatten_constr(gurobi_model, input_layer, start_dim, end_dim, name):

    if end_dim < 0:
        end_dim = input_layer.ndim + end_dim

    flatten_shape = 1
    for dim in range(start_dim, end_dim + 1):
        flatten_shape *= input_layer.shape[dim]

    new_shape = input_layer.shape[:start_dim] + (flatten_shape,) + input_layer.shape[end_dim + 1:]

    return input_layer.reshape(new_shape)

