# %%
import multiprocessing
import os

import gpytorch
import h5py
import numpy as np
import torch
import tqdm.auto as tqdm
from joblib import Parallel, delayed

from dgp_bo import DATA_DIR
from dgp_bo.dirichlet import dirichlet5D
from dgp_bo.gp import ExactGPModel
from dgp_bo.multiobjective import EHVI, HV_Calc, Pareto_finder
from dgp_bo.utils import lookup_in_h5_file

# %%
SMOKE_TEST = True
TRAINING_ITERATIONS = 2 if SMOKE_TEST else 500
BO_ITERATIONS = 10 if SMOKE_TEST else 500

file_out = os.path.basename(__file__)[:-3]
MOR_FILENAME = os.path.join(DATA_DIR, "5space-mor.h5")
TEC_FILENAME = os.path.join(DATA_DIR, "5space-md16-tec-new.h5")

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.Interval(0.001, 10)
)

# Lists for storing results
gp_mean_tec_lst = []
gp_mean_bulk_lst = []
gp_std_tec_lst = []
gp_std_bulk_lst = []

ref = np.array([[180, 0]])
goal = np.array([[0, 1]])

x_eval = torch.from_numpy((dirichlet5D(2)) / 32)
y_tec = lookup_in_h5_file(x_eval, TEC_FILENAME, "tec")
y_bulk = lookup_in_h5_file(x_eval, MOR_FILENAME, "bulkmodul_eq")

# Train thermal expansion coefficient model
likelihood_tec = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.Interval(0.001, 10)
)
model_tec = ExactGPModel(x_eval, y_tec, likelihood_tec)
model_tec.train()
likelihood_tec.train()
optimizer_tec = torch.optim.Adam(model_tec.parameters(), lr=0.1)
mll_tec = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_tec, model_tec)
for _i in range(TRAINING_ITERATIONS):
    optimizer_tec.zero_grad()
    output_tec = model_tec(x_eval)
    loss_tec = -mll_tec(output_tec, y_tec)
    loss_tec.backward()
    optimizer_tec.step()

model_tec.eval()
likelihood_tec.eval()

# Train bulk modulus model
likelihood_bulk = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.Interval(0.001, 10)
)
model_bulk = ExactGPModel(x_eval, y_bulk, likelihood_bulk)
model_bulk.train()
likelihood_bulk.train()
optimizer_bulk = torch.optim.Adam(model_bulk.parameters(), lr=0.1)
mll_bulk = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_bulk, model_bulk)
for _i in range(TRAINING_ITERATIONS):
    optimizer_bulk.zero_grad()
    output_bulk = model_bulk(x_eval)
    loss_bulk = -mll_bulk(output_bulk, y_bulk)
    loss_bulk.backward()
    optimizer_bulk.step()
model_bulk.eval()
likelihood_bulk.eval()

yp = np.concatenate(
    (y_tec.reshape(y_tec.shape[0], 1), y_bulk.reshape(y_bulk.shape[0], 1)), axis=1
)
yp_query = yp.reshape(y_tec.shape[0], 2)

data_yp = yp_query
y_pareto_truth, ind = Pareto_finder(data_yp, goal)
hv_truth = (HV_Calc(goal, ref, y_pareto_truth)).reshape(1, 1)

for _k in tqdm.tqdm(range(BO_ITERATIONS)):
    torch.cuda.empty_cache()
    torch.set_flush_denormal(True)
    xi = 0.3

    sampled_x = torch.from_numpy((dirichlet5D(500)) / 32)
    model_tec.eval()
    likelihood_tec.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred_tec = likelihood_tec(model_tec(sampled_x))
        lower_tec, upper_tec = observed_pred_tec.confidence_region()
        mean_tec = observed_pred_tec.mean.detach()
        var_tec = (upper_tec.detach() - mean_tec) / 2

    model_bulk.eval()
    likelihood_bulk.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred_bulk = likelihood_bulk(model_bulk(sampled_x))
        lower_bulk, upper_bulk = observed_pred_bulk.confidence_region()
        mean_bulk = observed_pred_bulk.mean.detach()
        var_bulk = (upper_bulk.detach() - mean_bulk) / 2

    eh_mean = np.concatenate(
        (np.array([mean_tec.numpy()]).T, np.array([mean_bulk.numpy()]).T), axis=1
    )
    eh_std = np.concatenate(
        (np.array([var_tec.numpy()]).T, np.array([var_bulk.numpy()]).T), axis=1
    )

    gp_mean_tec_lst.append(mean_tec.numpy())
    gp_mean_bulk_lst.append(mean_bulk.numpy())
    gp_std_tec_lst.append(var_tec.numpy())
    gp_std_bulk_lst.append(var_bulk.numpy())
    n_jobs = multiprocessing.cpu_count()

    def calc(ii):
        return EHVI(eh_mean[ii], eh_std[ii], goal, ref, y_pareto_truth)

    ehvi = Parallel(n_jobs)(delayed(calc)([jj]) for jj in range(eh_mean.shape[0]))
    ehvi = np.array(ehvi)

    x_star = np.argmax(ehvi)
    new_x = sampled_x.detach()[x_star]
    x_eval = torch.tensor(
        np.concatenate((x_eval.numpy(), new_x.unsqueeze(0).numpy()), axis=0)
    )

    new_y_tec = lookup_in_h5_file(new_x.unsqueeze(0), TEC_FILENAME, "tec")
    new_y_bulk = lookup_in_h5_file(new_x.unsqueeze(0), MOR_FILENAME, "bulkmodul_eq")
    y_tec = torch.tensor(np.concatenate((y_tec, new_y_tec.numpy()), axis=0))
    y_bulk = torch.tensor(np.concatenate((y_bulk, new_y_bulk.numpy()), axis=0))

    yp = np.concatenate(
        (y_tec.reshape(y_tec.shape[0], 1), y_bulk.reshape(y_bulk.shape[0], 1)), axis=1
    )
    yp_query = yp.reshape(yp.shape[0], 2)
    data_yp = yp

    y_pareto_truth, ind = Pareto_finder(data_yp, goal)
    hv_t = (HV_Calc(goal, ref, y_pareto_truth)).reshape(1, 1)
    hv_truth = np.concatenate((hv_truth, hv_t.reshape(1, 1)))

    # Train thermal expansion coefficient model
    model_tec = ExactGPModel(x_eval, y_tec, likelihood_tec)
    model_tec.train()
    likelihood_tec.train()
    optimizer_tec = torch.optim.Adam(model_tec.parameters(), lr=0.1)
    mll_tec = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_tec, model_tec)

    for _i in range(TRAINING_ITERATIONS):
        optimizer_tec.zero_grad()
        output_tec = model_tec(x_eval)
        loss_tec = -mll_tec(output_tec, y_tec)
        loss_tec.backward()
        optimizer_tec.step()

    model_tec.eval()
    likelihood_tec.eval()

    # Train bulk modulus model
    model_bulk = ExactGPModel(x_eval, y_bulk, likelihood_bulk)
    model_bulk.train()
    likelihood_bulk.train()
    optimizer_bulk = torch.optim.Adam(model_bulk.parameters(), lr=0.1)
    mll_bulk = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_bulk, model_bulk)

    for _i in range(TRAINING_ITERATIONS):
        optimizer_bulk.zero_grad()
        output_bulk = model_bulk(x_eval)
        loss_bulk = -mll_bulk(output_bulk, y_bulk)
        loss_bulk.backward()
        optimizer_bulk.step()

    model_bulk.eval()
    likelihood_bulk.eval()


# %%
with h5py.File(file_out + ".h5", "w") as f:
    f.create_dataset("hv_truth", data=hv_truth)
    f.create_dataset("x_eval", data=x_eval)
    f.create_dataset("y_tec", data=y_tec)
    f.create_dataset("y_bulk", data=y_bulk)
    f.create_dataset("gp_mean_tec", data=gp_mean_tec_lst)
    f.create_dataset("gp_mean_bulk", data=gp_mean_bulk_lst)
    f.create_dataset("gp_std_tec", data=gp_std_tec_lst)
    f.create_dataset("gp_std_bulk", data=gp_std_bulk_lst)
