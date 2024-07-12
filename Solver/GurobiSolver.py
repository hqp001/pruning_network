import time

import torch
import torchvision
import numpy as np

import gurobipy as gp
#from gurobi_ml import add_predictor_constr
from .torch2gurobi import add_predictor_constr

def dense_passing(model, where):

    if where == gp.GRB.Callback.MIPSOL:

        print("Found a solution")

        input_sol = model.cbGetSolution(model._input)

        dense_output = torch.argmax(model._network.forward(torch.tensor(input_sol, dtype=torch.float32))).item()

        if dense_output != model._correct_label:
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

        print("Dense output: ", dense_output, model._correct_label)

        if dense_output != model._correct_label:
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


def solve_with_gurobi(nn_regression, dense_model, image_range, correct_label, wrong_label, time_limit, callback):

    lb_image = image_range[0]
    ub_image = image_range[1]

    m = gp.Model()
    m.setParam('OutputFlag', 1)
    m.setParam('TimeLimit', time_limit)

    x = m.addMVar(lb_image.shape, lb=lb_image, ub=ub_image, name="x")
    y = m.addMVar((1, 10), lb=-gp.GRB.INFINITY, name="y")

    add_predictor_constr(m, nn_regression, x, y)

    m.setObjective(y[0][wrong_label] - y[0][correct_label], gp.GRB.MAXIMIZE)

    m._network = dense_model
    m._correct_label = correct_label
    m._x_max_sol = None
    m._input = x

    if callback == "none":
        # Add these contraints if neccessary
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


    if m.status == 4:
        print(f'Model is infeasible or unbounded')
        return None, None, m.Runtime

    elif m.status == gp.GRB.OPTIMAL or m.SolCount > 0:
        values = x.X

        if m._x_max_sol is None:
            m._x_max_sol = values

        return m._x_max_sol, m.ObjVal, m.Runtime

    elif m.status == gp.GRB.INTERRUPTED or m.status == gp.GRB.TIME_LIMIT:
        return m._x_max_sol, m.ObjVal, m.Runtime

    else:
        raise ValueError(f"Unexpected status : {m.status}")

