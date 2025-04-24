# %%
import multiprocessing
import os

import gpytorch
import h5py
import numpy as np
import torch
import tqdm.auto as tqdm
from gpytorch import settings
from joblib import Parallel, delayed

from dgp_bo import DATA_DIR
from dgp_bo.dirichlet import dirichlet5D
from dgp_bo.mtgp import MultitaskGPModel
from dgp_bo.multiobjective import EHVI, HV_Calc, Pareto_finder
from dgp_bo.utils import lookup_in_h5_file

# %%
SMOKE_TEST = True
TRAINING_ITERATIONS = 2 if SMOKE_TEST else 500
BO_ITERATIONS = 10 if SMOKE_TEST else 500

file_out = os.path.basename(__file__)[:-3]
MOR_FILENAME = os.path.join(DATA_DIR, "5space-mor.h5")
TEC_FILENAME = os.path.join(DATA_DIR, "5space-md16-tec-new.h5")

mtgp_mean1_lst = []
mtgp_mean2_lst = []
mtgp_mean3_lst = []
mtgp_std1_lst = []
mtgp_std2_lst = []
mtgp_std3_lst = []

ref = np.array([[180, 0]])
goal = np.array([[0, 1]])
settings.debug.off()

train_x1 = torch.from_numpy((dirichlet5D(2)) / 32)
train_x2 = train_x1
train_x3 = train_x1

train_y1 = lookup_in_h5_file(train_x1, TEC_FILENAME, "tec")
train_y2 = lookup_in_h5_file(train_x2, MOR_FILENAME, "bulkmodul_eq")
train_y3 = lookup_in_h5_file(train_x3, MOR_FILENAME, "volume_eq")

likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.Interval(0.001, 10)
)

train_i_task1 = torch.full((train_x1.shape[0], 1), dtype=torch.long, fill_value=0)
train_i_task2 = torch.full((train_x2.shape[0], 1), dtype=torch.long, fill_value=1)
train_i_task3 = torch.full((train_x3.shape[0], 1), dtype=torch.long, fill_value=2)

full_train_x = torch.cat([train_x1, train_x2, train_x3])
full_train_i = torch.cat([train_i_task1, train_i_task2, train_i_task3])
full_train_y = torch.cat([train_y1, train_y2, train_y3])

model = MultitaskGPModel((full_train_x, full_train_i), full_train_y, likelihood)
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for _i in range(TRAINING_ITERATIONS):
    optimizer.zero_grad()
    output = model(full_train_x, full_train_i)
    loss = -mll(output, full_train_y)
    loss.backward()
    optimizer.step()

model.eval()
likelihood.eval()

yp = np.concatenate(
    (train_y1.reshape(train_y1.shape[0], 1), train_y2.reshape(train_y2.shape[0], 1)),
    axis=1,
)
yp_query = yp.reshape(train_y1.shape[0], 2)

data_yp = yp_query

y_pareto_truth, ind = Pareto_finder(data_yp, goal)
hv_truth = (HV_Calc(goal, ref, y_pareto_truth)).reshape(1, 1)


# %%
test_x = torch.from_numpy((dirichlet5D(500)) / 32)
for _k in tqdm.tqdm(range(BO_ITERATIONS)):
    torch.cuda.empty_cache()
    torch.set_flush_denormal(True)
    xi = 0.9
    test_x = torch.from_numpy((dirichlet5D(500)) / 32)
    test_i_task1 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=0)
    test_i_task2 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=1)
    test_i_task3 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=2)

    test_i1 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=0)

    test_input1 = torch.cat((test_i1, test_x), dim=-1)

    test_i2 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=1)

    test_input2 = torch.cat((test_i2, test_x), dim=-1)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred_y1 = likelihood(model(test_x, test_i_task1))
        observed_pred_y2 = likelihood(model(test_x, test_i_task2))
        observed_pred_y3 = likelihood(model(test_x, test_i_task3))

    upper1, lower1 = observed_pred_y1.confidence_region()
    std1 = (lower1.numpy() - observed_pred_y1.mean.detach().numpy()) / 2
    mean_t1 = observed_pred_y1.mean.detach()
    var_t1 = torch.tensor(std1)
    upper2, lower2 = observed_pred_y2.confidence_region()
    std2 = (lower2.numpy() - observed_pred_y2.mean.detach().numpy()) / 2
    mean_t2 = observed_pred_y2.mean.detach()
    var_t2 = torch.tensor(std2)

    upper3, lower3 = observed_pred_y3.confidence_region()
    std3 = (lower3.numpy() - observed_pred_y3.mean.detach().numpy()) / 2
    mean_t3 = observed_pred_y3.mean.detach()
    var_t3 = torch.tensor(std3)

    mtgp_mean1_lst.append(mean_t1.numpy())
    mtgp_mean2_lst.append(mean_t2.numpy())
    mtgp_mean3_lst.append(mean_t3.numpy())
    mtgp_std1_lst.append(var_t1.numpy())
    mtgp_std2_lst.append(var_t2.numpy())
    mtgp_std3_lst.append(var_t3.numpy())
    eh_mean = np.concatenate(
        (np.array([mean_t1.numpy()]).T, np.array([mean_t2.numpy()]).T), axis=1
    )
    eh_std = np.concatenate(
        (np.array([var_t1.numpy()]).T, np.array([var_t2.numpy()]).T), axis=1
    )
    n_jobs = multiprocessing.cpu_count()

    def calc(ii):
        return EHVI(eh_mean[ii], eh_std[ii], goal, ref, y_pareto_truth)

    ehvi = Parallel(n_jobs)(delayed(calc)([jj]) for jj in range(eh_mean.shape[0]))
    ehvi = np.array(ehvi)

    x_star = np.argmax(ehvi)

    new_x = test_x.detach()[x_star]
    test_x = torch.cat((test_x[:x_star], test_x[x_star + 1 :]))
    new_y = lookup_in_h5_file(new_x.unsqueeze(0), MOR_FILENAME, "bulkmodul_eq")
    new_y2 = lookup_in_h5_file(new_x.unsqueeze(0), MOR_FILENAME, "volume_eq")

    data_x = np.concatenate((train_x2.numpy(), new_x.unsqueeze(0).numpy()), axis=0)
    data_y = np.concatenate((train_y2, new_y.numpy()), axis=0)
    train_x2 = torch.tensor(data_x)
    train_y2 = torch.tensor(data_y)

    data_x = np.concatenate((train_x3.numpy(), new_x.unsqueeze(0).numpy()), axis=0)
    data_y = np.concatenate((train_y3, new_y2.numpy()), axis=0)
    train_x3 = torch.tensor(data_x)
    train_y3 = torch.tensor(data_y)

    new_y1 = lookup_in_h5_file(new_x.unsqueeze(0), TEC_FILENAME, "tec")
    data_x = np.concatenate((train_x1.numpy(), new_x.unsqueeze(0).numpy()), axis=0)
    data_y = np.concatenate((train_y1, new_y1.numpy()), axis=0)
    train_x1 = torch.tensor(data_x)
    train_y1 = torch.tensor(data_y)

    train_i_task1 = torch.full((train_x1.shape[0], 1), dtype=torch.long, fill_value=0)
    train_i_task2 = torch.full((train_x2.shape[0], 1), dtype=torch.long, fill_value=1)
    train_i_task3 = torch.full((train_x3.shape[0], 1), dtype=torch.long, fill_value=2)

    full_train_x = torch.cat([train_x1, train_x2, train_x3])
    full_train_i = torch.cat([train_i_task1, train_i_task2, train_i_task3])
    full_train_y = torch.cat([train_y1, train_y2, train_y3])

    yp = np.concatenate(
        (
            train_y1.reshape(train_y1.shape[0], 1),
            train_y2.reshape(train_y2.shape[0], 1),
        ),
        axis=1,
    )
    yp_query = yp.reshape(yp.shape[0], 2)
    data_yp = yp

    y_pareto_truth, ind = Pareto_finder(data_yp, goal)
    hv_t = (HV_Calc(goal, ref, y_pareto_truth)).reshape(1, 1)
    hv_truth = np.concatenate((hv_truth, hv_t.reshape(1, 1)))

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Interval(0.001, 10)
    )
    model = MultitaskGPModel((full_train_x, full_train_i), full_train_y, likelihood)

    training_iterations = 2 if SMOKE_TEST else 500

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _i in range(TRAINING_ITERATIONS):
        optimizer.zero_grad()
        output = model(full_train_x, full_train_i)
        loss = -mll(output, full_train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    likelihood.eval()

# %%
with h5py.File(file_out + ".h5", "w") as f:
    f.create_dataset("hv_truth", data=hv_truth)

    # Training data
    train_data = f.create_group("train")
    train_data.create_dataset("x1", data=train_x1)
    train_data.create_dataset("x2", data=train_x2)
    train_data.create_dataset("x3", data=train_x3)
    train_data.create_dataset("y1", data=train_y1)
    train_data.create_dataset("y2", data=train_y2)
    train_data.create_dataset("y3", data=train_y3)

    # MTGP predictions
    predictions = f.create_group("predictions")
    predictions.create_dataset("mean1", data=mtgp_mean1_lst)
    predictions.create_dataset("mean2", data=mtgp_mean2_lst)
    predictions.create_dataset("mean3", data=mtgp_mean3_lst)
    predictions.create_dataset("std1", data=mtgp_std1_lst)
    predictions.create_dataset("std2", data=mtgp_std2_lst)
    predictions.create_dataset("std3", data=mtgp_std3_lst)
