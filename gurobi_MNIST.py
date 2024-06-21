import numpy as np
from matplotlib import pyplot as plt
import time

import torch
import torchvision
from skorch import NeuralNetClassifier

import gurobipy as gp
from gurobipy import GRB
from gurobi_ml import add_predictor_constr

def dense_passing(model, where):

    if where == GRB.Callback.MIPSOL:
        variables = model.getVars()

        input_variables = [var for var in variables if var.VarName.startswith("x")]

        input_sol = [model.cbGetSolution(var) for var in input_variables]

        dense_output = torch.argmax(model._network.forward(torch.tensor(input_sol)))

        if dense_output != model._right_label:
            print("Stopping the optimization...")
            model._x_max_sol = input_sol
            model.terminate()


def solve_optimal_adversary_with_gurobi(nn_regression, dense_model, image, wrong_label, right_label, delta, ex_prob, timelimit, callback = "none"):

    m = gp.Model()

    m._network = dense_model
    m._right_label = torch.argmax(m._network.forward(torch.tensor(image)))
    m._x_max_sol = None


    x = m.addMVar(image.shape, lb=0.0, ub=1.0, name="x")
    y = m.addMVar(ex_prob.detach().numpy().shape, lb=-gp.GRB.INFINITY, name="y")

    abs_diff = m.addMVar(image.shape, lb=0, ub=1, name="abs_diff")

    m.setObjective(y[wrong_label] - y[right_label], gp.GRB.MAXIMIZE)

    # Bound on the distance to example in norm-1
    m.addConstr(abs_diff >= x - image)
    m.addConstr(abs_diff >= -x + image)
    m.addConstr(abs_diff.sum() <= delta)

    pred_constr = add_predictor_constr(m, nn_regression, x, y)
    pred_constr.print_stats()

    m.setParam('TimeLimit', timelimit)
    m.setParam('OutputFlag', 1)
    m.Params.BestBdStop = 0.0
    m.Params.BestObjStop = 0.0

    if callback == "none":

        m.optimize()

    elif callback == "dense_passing":

        m.optimize(dense_passing)

    else:

        print("Wrong callback name")

        return None, None, None


    # Calculate the best gap (optimality gap)
    # best_gap = None
    # if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
    #     best_gap = 0 # m.MIPGap

    if m.status == 4:
        return None, None, m.Runtime
    elif m.status == GRB.OPTIMAL:
        values = x.X.reshape(-1).tolist()
        if m._x_max_sol == None:
            m._x_max_sol = values

        return m._x_max_sol, y[wrong_label].x - y[right_label].x, m.Runtime
    elif m.SolCount > 0:
        m.setParam(GRB.Param.SolutionNumber, 0)
        output = m.PoolObjVal
        values = x.Xn.reshape(-1).tolist()

        if m._x_max_sol == None:
            m._x_max_sol = values

        return m._x_max_sol, output, m.Runtime
    else:
        print(f'end with status: {m.status}')
        return None, None, m.Runtime


def solve_with_gurobi(nn_regression, dense_model, image, delta, ex_prob, wrong_label, right_label, time_limit, callback):
    x_max, max_, time_count = solve_optimal_adversary_with_gurobi(nn_regression, dense_model, image, wrong_label, right_label, delta, ex_prob, time_limit, callback)
    return x_max, max_, time_count
