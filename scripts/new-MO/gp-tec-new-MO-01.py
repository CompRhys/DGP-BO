# %%
import multiprocessing
import os

import gpytorch
import h5py
import numpy as np
import torch
from joblib import Parallel, delayed

from dgp_bo import DATA_DIR
from dgp_bo.dirichlet import dirichlet5D
from dgp_bo.gp import ExactGPModel
from dgp_bo.multiobjective import EHVI, HV_Calc, Pareto_finder
from dgp_bo.utils import bmft, cft

# %%
SMOKE_TEST = "CI" in os.environ
TRAINING_ITERATIONS = 2 if SMOKE_TEST else 500
BO_ITERATIONS = 10 if SMOKE_TEST else 500

file_out = os.path.basename(__file__)[:-3]
MOR_FILENAME = os.path.join(DATA_DIR, "5space-mor.h5")
TEC_FILENAME = os.path.join(DATA_DIR, "5space-md16-tec-new.h5")

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.Interval(0.001, 10)
)


# import tqdm
gp_mean1_lst = []
gp_mean2_lst = []
gp_std1_lst = []
gp_std2_lst = []

ref = np.array([[180, 0]])
goal = np.array([[0, 1]])

x1 = torch.from_numpy((dirichlet5D(2)) / 32)
x2 = x1
y1 = bmft(x1, TEC_FILENAME)
y2 = cft(x1, MOR_FILENAME)

# initialize likelihood and model
likelihood1 = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.Interval(0.001, 10)
)
likelihood2 = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.Interval(0.001, 10)
)
model1 = ExactGPModel(x1, y1, likelihood1)
model2 = ExactGPModel(x2, y2, likelihood2)

model1.train()
likelihood1.train()

optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.1)

mll1 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood1, model1)

for _i in range(TRAINING_ITERATIONS):
    optimizer1.zero_grad()
    output1 = model1(x1)
    loss1 = -mll1(output1, y1)
    loss1.backward()
    optimizer1.step()

model1.eval()
likelihood1.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x1 = torch.from_numpy((dirichlet5D(500)) / 32)
    observed_pred1 = likelihood(model1(test_x1))


model2.train()
likelihood2.train()

optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.1)

mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood2, model2)

for _i in range(TRAINING_ITERATIONS):
    optimizer2.zero_grad()
    output2 = model2(x2)
    loss2 = -mll2(output2, y2)
    loss2.backward()
    optimizer2.step()

model2.eval()
likelihood2.eval()
test_x1 = torch.from_numpy((dirichlet5D(500)) / 32)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred1 = likelihood1(model1(test_x1))

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred2 = likelihood2(model2(test_x1))

mean_tp1 = observed_pred1.mean.detach()
mean_tp2 = observed_pred2.mean.detach()
yp = np.concatenate((y1.reshape(y1.shape[0], 1), y2.reshape(y2.shape[0], 1)), axis=1)
yp_query = yp.reshape(y1.shape[0], 2)

data_yp = yp_query
y_pareto_truth, ind = Pareto_finder(data_yp, goal)
hv_truth = (HV_Calc(goal, ref, y_pareto_truth)).reshape(1, 1)

test_x = torch.from_numpy((dirichlet5D(500)) / 32)
test_x2 = test_x
test_x3 = test_x
test_x4 = test_x
for _k in range(BO_ITERATIONS):
    torch.cuda.empty_cache()
    torch.set_flush_denormal(True)
    xi = 0.3

    test_x = torch.from_numpy((dirichlet5D(500)) / 32)
    model1.eval()
    likelihood1.eval()

    for i in range(int(test_x.shape[0] / 5)):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            ind = i * 5
            if i == 0:
                test_xt = test_x[0:5, :]
                observed_pred_y1 = likelihood1(model1(test_xt.float()))

                lower1, upper1 = observed_pred_y1.confidence_region()
                mean1 = observed_pred_y1.mean.detach()

                mean_t1 = mean1
                var_t1 = (upper1.detach() - mean1) / 2
            else:
                test_xt = test_x[ind : ind + 5, :]
                observed_pred_y1 = likelihood1(model1(test_xt.float()))

                lower1, upper1 = observed_pred_y1.confidence_region()
                mean1 = observed_pred_y1.mean.detach()
                var1 = (upper1.detach() - mean1) / 2
                mean_t1 = torch.cat((mean_t1, mean1), 0)
                var_t1 = torch.cat((var_t1, var1), 0)

    test_xt = test_x[ind + 5 :, :]
    if test_xt.numel() != 0:
        observed_pred_y1 = likelihood1(model1(test_xt.float()))

        lower1, upper1 = observed_pred_y1.confidence_region()
        mean1 = observed_pred_y1.mean.detach()
        var1 = (upper1.detach() - mean1) / 2
        mean_t1 = torch.cat((mean_t1, mean1), 0)
        var_t1 = torch.cat((var_t1, var1), 0)

    model2.eval()
    likelihood2.eval()

    for i in range(int(test_x.shape[0] / 5)):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            ind = i * 5
            if i == 0:
                test_xt = test_x[0:5, :]

                observed_pred_y2 = likelihood2(model2(test_xt.float()))

                lower2, upper2 = observed_pred_y2.confidence_region()
                mean2 = observed_pred_y2.mean.detach()
                mean_t2 = mean2
                var_t2 = (upper2.detach() - mean2) / 2
            else:
                test_xt = test_x[ind : ind + 5, :]
                observed_pred_y2 = likelihood2(model2(test_xt.float()))

                lower2, upper2 = observed_pred_y2.confidence_region()
                mean2 = observed_pred_y2.mean.detach()
                var2 = (upper2.detach() - mean2) / 2
                mean_t2 = torch.cat((mean_t2, mean2), 0)
                var_t2 = torch.cat((var_t2, var2), 0)

    test_xt = test_x[ind + 5 :, :]
    if test_xt.numel() != 0:
        observed_pred_y2 = likelihood2(model2(test_xt.float()))

        lower2, upper2 = observed_pred_y2.confidence_region()
        mean2 = observed_pred_y2.mean.detach()
        var2 = (upper2.detach() - mean2) / 2
        mean_t2 = torch.cat((mean_t2, mean2), 0)
        var_t2 = torch.cat((var_t2, var2), 0)

    eh_mean = np.concatenate(
        (np.array([mean_t1.numpy()]).T, np.array([mean_t2.numpy()]).T), axis=1
    )
    eh_std = np.concatenate(
        (np.array([var_t1.numpy()]).T, np.array([var_t2.numpy()]).T), axis=1
    )

    gp_mean1_lst.append(mean_t1.numpy())
    gp_mean2_lst.append(mean_t2.numpy())
    gp_std1_lst.append(var_t1.numpy())
    gp_std2_lst.append(var_t2.numpy())
    n_jobs = multiprocessing.cpu_count()

    def calc(ii):
        return EHVI(eh_mean[ii], eh_std[ii], goal, ref, y_pareto_truth)

    ehvi = Parallel(n_jobs)(delayed(calc)([jj]) for jj in range(eh_mean.shape[0]))
    ehvi = np.array(ehvi)

    x_star = np.argmax(ehvi)

    new_x = test_x.detach()[x_star]

    new_y1 = bmft(new_x.unsqueeze(0), TEC_FILENAME)
    data_x = np.concatenate((x1.numpy(), new_x.unsqueeze(0).numpy()), axis=0)
    data_y = np.concatenate((y1, new_y1.numpy()), axis=0)

    x1 = torch.tensor(data_x)
    y1 = torch.tensor(data_y)

    new_y2 = cft(new_x.unsqueeze(0), MOR_FILENAME)
    data_x = np.concatenate((x2.numpy(), new_x.unsqueeze(0).numpy()), axis=0)
    data_y = np.concatenate((y2, new_y2.numpy()), axis=0)

    x2 = torch.tensor(data_x)
    y2 = torch.tensor(data_y)

    yp = np.concatenate((y1.reshape(y1.shape[0], 1), y2.reshape(y1.shape[0], 1)), axis=1)
    yp_query = yp.reshape(yp.shape[0], 2)
    data_yp = yp

    y_pareto_truth, ind = Pareto_finder(data_yp, goal)
    hv_t = (HV_Calc(goal, ref, y_pareto_truth)).reshape(1, 1)
    hv_truth = np.concatenate((hv_truth, hv_t.reshape(1, 1)))

    model1 = ExactGPModel(x1, y1, likelihood1)
    model2 = ExactGPModel(x2, y2, likelihood2)

    model1.train()
    likelihood1.train()

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.1)

    mll1 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood1, model1)

    for _i in range(TRAINING_ITERATIONS):
        optimizer1.zero_grad()
        output1 = model1(x1)
        loss1 = -mll1(output1, y1)
        loss1.backward()
        optimizer1.step()

    model1.eval()
    likelihood1.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x1 = torch.from_numpy((dirichlet5D(500)) / 32)
        observed_pred1 = likelihood(model1(test_x1))

    model2.train()
    likelihood2.train()

    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.1)

    mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood2, model2)

    for _i in range(TRAINING_ITERATIONS):
        optimizer2.zero_grad()
        output2 = model2(x2)
        loss2 = -mll2(output2, y2)
        loss2.backward()
        optimizer2.step()

    model2.eval()
    likelihood2.eval()


# %%
with h5py.File(file_out + ".h5", "w") as f:
    f.create_dataset("hv_truth", data=hv_truth)
    f.create_dataset("x1", data=x1)
    f.create_dataset("x2", data=x2)
    f.create_dataset("y1", data=y1)
    f.create_dataset("y2", data=y2)
    f.create_dataset("gp_mean1", data=gp_mean1_lst)
    f.create_dataset("gp_mean2", data=gp_mean2_lst)
    f.create_dataset("gp_std1", data=gp_std1_lst)
    f.create_dataset("gp_std2", data=gp_std2_lst)
