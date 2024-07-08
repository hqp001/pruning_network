import numpy as np
import time

import torch
import torchvision

import gurobipy as gp
from gurobipy import GRB
#from gurobi_ml import add_predictor_constr
from torch2gurobi import add_predictor_constr

import time

# from mnist_io_csv import *
# from mnist_util import *


def dense_passing(model, where):

    if where == GRB.Callback.MIPSOL:

        print("Found a solution")

        variables = model.getVars()

        input_variables = [var for var in variables if var.VarName.startswith("x")]

        input_sol = [model.cbGetSolution(var) for var in input_variables]

        dense_output = torch.argmax(model._network.forward(torch.tensor(input_sol))).item()

        if dense_output != model._right_label:
            print("Stopping the optimization...")
            model._x_max_sol = input_sol
            model.terminate()

def remove_region(model, where):

    if where == gp.GRB.Callback.MIPSOL:

        print("Found a solution")

        variables = model.getVars()

        input_variables = [var for var in variables if var.VarName.startswith("x")]

        input_sol = [model.cbGetSolution(var) for var in input_variables]

        dense_output = torch.argmax(model._network.forward(torch.tensor(input_sol))).item()

        print("Dense output: ", dense_output, model._right_label)

        if dense_output != model._right_label:
            model._x_max_sol = input_sol
            print("Terminating the model")
            model.terminate()

        neuron_variables = [var for var in variables if var.VarName.startswith("neuron")]


        neuron_sol = [model.cbGetSolution(var) for var in neuron_variables]

        added_constr = 0

        #print(neuron_variables)

        for i in range(len(neuron_sol)):
            if (neuron_sol[i] <= 0.5):
                added_constr += neuron_variables[i]
            else:
                added_constr += (1 - neuron_variables[i])

        #print(added_constr)

        model.cbLazy(added_constr >= 1)

        print("Removed region")





def solve_optimal_adversary_with_gurobi(nn_regression, dense_model, image_range, correct_label, wrong_label, time_limit, callback):

    #nn_regression = torch.nn.Sequential(*nn_regression.layers)
    m = gp.Model()

    m._network = dense_model
    m._right_label = correct_label
    m._x_max_sol = None


    m.setParam('OutputFlag', 1)

    x = m.addMVar(len(image_range), name="x")
    y = m.addMVar((10, ), lb=-gp.GRB.INFINITY, name="y")
    add_predictor_constr(m, nn_regression, x, y)
    # pred_constr.print_stats()


    for i in range(len(image_range)):

        # print(image_range[i])
        x[i].ub = image_range[i][1]
        x[i].lb = image_range[i][0]

    print(wrong_label, correct_label)


    m.setObjective(y[wrong_label] - y[correct_label], gp.GRB.MAXIMIZE)

    m.setParam('TimeLimit', time_limit)
    # return None, None, None

    if callback == "none":

        # m.Params.BestBdStop = 0.0
        # m.Params.BestObjStop = 0.0

        m.optimize()

    elif callback == "dense_passing":

        m.optimize(dense_passing)

    elif callback == "remove_region":

        m.setParam("LazyConstraints", 1)

        m.optimize(remove_region)

    else:

        raise ValueError(f"Callback {callback} doesn't exist")


    # Calculate the best gap (optimality gap)
    # best_gap = None
    # if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
    #     best_gap = 0 # m.MIPGap

    if m.status == 4:
        # print(f'{info} is infeasible or unbounded')
        return None, None, m.Runtime
    elif m.status == GRB.OPTIMAL or m.SolCount > 0:
        # values = [x[i].x for i in range(len(x.tolist()))]

        values = x.X.reshape(-1).tolist()

        if m._x_max_sol == None:
            m._x_max_sol = values

        return m._x_max_sol, m.ObjVal, m.Runtime

    elif m.status == GRB.INTERRUPTED or m.status == GRB.TIME_LIMIT:

        return m._x_max_sol, m.ObjVal, m.Runtime

    else:
        #print(f'end with status: {m.status}')
        raise ValueError(f"Unexpected status : {m.status}")


def solve_with_gurobi_and_record(surrogate_model, origin_model, image_range, output_range, time_limit, correct_label, wrong_label, callback):


    #print(correct_label, wrong_label)

    x_max, max_, time_count = solve_optimal_adversary_with_gurobi(surrogate_model, origin_model, image_range, correct_label, wrong_label, time_limit, callback)

    # print(max_, time_count)

    return x_max, max_, time_count
    # input_size, layer_dims = get_model_info(nn_regression)
    # layer_num = len(layer_dims) - 1
    # if max_ is None:
    #     store_data_gurobi(
    #         [['GUROBI', [input_size] + layer_dims, seed, image_index, x_max, None,
    #           time_count,
    #           best_gap, None, right_label, wrong_label, info]], f'Gurobi_MNIST_Benchmark_{time_limit}_{info[-1]}.csv')
    # else:
    #     res = nn_regression(torch.FloatTensor(x_max)).tolist()
    #     store_data_gurobi(
    #         [['GUROBI', [input_size] + layer_dims, seed, image_index, x_max, res[wrong_label] - res[right_label], time_count,
    #           best_gap, res, right_label, wrong_label, info]], f'Gurobi_MNIST_Benchmark_{time_limit}_{info[-1]}.csv')
    # return x_max, max_, time_count
