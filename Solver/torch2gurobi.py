import torch
import numpy as np
import gurobipy as gp

def add_predictor_constr(gurobi_model, nn_model, x, y):

    sequential = []

    sequential.append(x)

    for idx, layer in enumerate(nn_model.layers):
        if (isinstance(layer, torch.nn.Linear)):
            linear = add_linear_constr2(gurobi_model, sequential[-1], layer, name = f"linear_{idx}")

            sequential.append(linear)

        elif (isinstance(layer, torch.nn.ReLU)):
            relu = add_relu_constr2(gurobi_model, sequential[-1], name = f"relu_{idx}")

            sequential.append(relu)

        elif (isinstance(layer, torch.nn.Conv2d)):
            conv = add_conv2d_constr(gurobi_model, sequential[-1], layer, name = f"conv2d_{idx}")

            sequential.append(conv)

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

def add_conv2d_constr(gurobi_model, input_layer, conv_layer, name = "conv2d"):

    # Assert batch size is 1
    assert input_layer.shape[0] == 1
    assert conv_layer.padding_mode == "zeros"

    out_channels = conv_layer.out_channels
    in_channels = conv_layer.in_channels
    kernel_size = conv_layer.kernel_size
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation

    weight = conv_layer.weight
    bias = conv_layer.bias

    dummy_output = conv_layer(torch.ones(input_layer.shape))

    in_size = (input_layer.shape[2], input_layer.shape[3])
    out_size = (dummy_output.shape[2], dummy_output.shape[3])

    unfolded_tensor = torch.nn.Unfold(kernel_size = conv_layer.kernel_size, padding = conv_layer.padding, stride = conv_layer.stride, dilation = conv_layer.dilation)

    def index_tensor_to_MVar(index_tensor, gurobi_var, zero_var):

        var_list = []

        gurobi_var = gurobi_var.tolist()

        for i in range(index_tensor.size):
            if index_tensor[i] == 0:
                var_list.append(zero_var)
            else:
                var_list.append(gurobi_var[index_tensor[i] - 1])

        return gp.MVar.fromlist(var_list)

    def convo(image, filter_weight, unfolded_tensor, zero_var):
        feature_map = []

        index_tensor = torch.arange(1, image.size + 1, dtype=torch.float32)

        unf = unfolded_tensor(index_tensor.view(1, image.shape[0], image.shape[1])).transpose(0, 1).int().numpy()

        for i in range(unf.shape[0]):

            gurobi_patch = index_tensor_to_MVar(unf[i], image.reshape(-1), zero_var)

            feature_pixel = gurobi_patch @ filter_weight.reshape(-1).T

            feature_map.append(feature_pixel)

        return feature_map

    zero_var = gurobi_model.addVar(lb=0, ub=0, name="zero")

    output_images = gurobi_model.addMVar((1, out_channels, out_size[0], out_size[1]), lb=-gp.GRB.INFINITY, name=name)
    output_images = output_images.reshape(1, out_channels, -1)

    for i in range(out_channels):
        feature_image = []

        for j in range(out_size[0] * out_size[1]):
            feature_image.append(bias[i].item())

        for j in range(in_channels):

            feature_map = convo(input_layer[0][j], weight[i][j].detach().numpy(), unfolded_tensor, zero_var)

            assert len(feature_map) == len(feature_image)

            for k in range(len(feature_image)):
                feature_image[k] += feature_map[k]

        for j in range(len(feature_image)):

            gurobi_model.addConstr(output_images[0][i][j] == feature_image[j])

    output_images = output_images.reshape(1, out_channels, out_size[0], out_size[1])

    return output_images

def add_linear_constr(gurobi_model, input_layer, linear_layer, name="linear"):

    input_size = linear_layer.in_features
    output_size = linear_layer.out_features

    weight = linear_layer.weight.data.numpy()
    bias = linear_layer.bias.data.numpy()

    assert(input_size == input_layer.size and output_size == bias.size)

    output_layer = ( input_layer @ weight.T ) + bias

    return output_layer

def add_linear_constr2(gurobi_model, input_layer, linear_layer, name="linear"):

    assert input_layer.shape[0] == 1
    assert len(input_layer.shape) == 2

    input_size = linear_layer.in_features
    output_size = linear_layer.out_features

    weight = linear_layer.weight.detach().numpy()
    bias = linear_layer.bias.detach().numpy()

    assert(input_size == input_layer.size and output_size == bias.size)

    output_layer = gurobi_model.addMVar((1, output_size), lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name=name)

    gurobi_model.addConstr(output_layer == ( input_layer @ weight.T ) + bias)

    return output_layer

def add_relu_constr2(gurobi_model, input_layer, name):

    assert input_layer.shape[0] == 1

    input_shape = input_layer.shape
    output_layer = gurobi_model.addMVar(input_shape, name=name)

    input_layer = input_layer.reshape(-1)
    output_layer = output_layer.reshape(-1)

    for i in range(input_layer.size):
       gurobi_model.addConstr(output_layer[i] == gp.max_([input_layer[i]], constant=0.0))

    input_layer = input_layer.reshape(input_shape)
    output_layer = output_layer.reshape(input_shape)

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

# Assert if gurobi_model formulate correctly
def check_correct_formulation(gurobi_model, nn_model, x, y):

    gurobi_model.update()

    input_shape = x.shape

    lower_bound = np.nan_to_num(x.lb.astype(np.float32), copy=True, nan=0.0, posinf=10, neginf=-10)
    upper_bound = np.nan_to_num(x.ub.astype(np.float32), copy=True, nan=0.0, posinf=10, neginf=10)
    lower_bound_tensor = torch.tensor(lower_bound, dtype=torch.float32)
    upper_bound_tensor = torch.tensor(upper_bound, dtype=torch.float32)

    random_input = torch.rand(*input_shape, dtype=torch.float32)
    random_input = lower_bound_tensor + (upper_bound_tensor - lower_bound_tensor) * random_input

    input_constr = gurobi_model.addConstr(x == random_input.numpy())

    current_objective = (gurobi_model.getObjective(), gurobi_model.ModelSense)

    gurobi_model.setObjective(y[0][0], gp.GRB.MAXIMIZE)
    gurobi_model.optimize()

    if gurobi_model.status != gp.GRB.OPTIMAL or gurobi_model.SolCount != 1:
        return False

    nn_model.eval()
    with torch.no_grad():
        output = nn_model(random_input).detach().numpy()

    gurobi_output = y.X

    print(gurobi_output)
    print(output)

    gurobi_model.remove(input_constr)
    gurobi_model.setObjective(current_objective[0], current_objective[1])
    gurobi_model.update()

    error = np.max(np.abs(output - gurobi_output))

    print("Diff: ", error)

    return error < 1e-5


