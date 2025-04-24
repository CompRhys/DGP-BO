# %%
import multiprocessing
import os
from copy import deepcopy

import gpytorch
import h5py
import numpy as np
import torch
import tqdm.auto as tqdm
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from joblib import Parallel, delayed

from dgp_bo import DATA_DIR
from dgp_bo.dirichlet import dirichlet5D
from dgp_bo.mtdgp import MultitaskIsoDeepGP
from dgp_bo.multiobjective import EHVI, HV_Calc, Pareto_finder
from dgp_bo.utils import lookup_in_h5_file

# %%
SMOKE_TEST = True
TRAINING_ITERATIONS = 2 if SMOKE_TEST else 500
BO_ITERATIONS = 10 if SMOKE_TEST else 500

file_out = os.path.basename(__file__)[:-3]
MOR_FILENAME = os.path.join(DATA_DIR, "5space-mor.h5")
TEC_FILENAME = os.path.join(DATA_DIR, "5space-md16-tec-new.h5")

num_tasks = 3
n_input_dims = 5
n_output_dims = 4

dgp_mean1_lst = []
dgp_mean2_lst = []
dgp_mean3_lst = []
dgp_std1_lst = []
dgp_std2_lst = []
dgp_std3_lst = []

ref = np.array([[180, 0]])
goal = np.array([[0, 1]])

x1 = torch.from_numpy((dirichlet5D(2)) / 32).to(dtype=torch.float32)
x2 = x1
x3 = x1


y1 = lookup_in_h5_file(x1, TEC_FILENAME, "tec")
y2 = lookup_in_h5_file(x2, MOR_FILENAME, "bulkmodul_eq")
y3 = lookup_in_h5_file(x3, MOR_FILENAME, "volume_eq")

full_train_y = torch.stack([y1, y2, y3], -1)

model = MultitaskIsoDeepGP(n_input_dims, n_output_dims, num_tasks).to(dtype=torch.float32)
likelihood = model.likelihood

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
mll = DeepApproximateMLL(
    VariationalELBO(likelihood, model, num_data=full_train_y.size(0))
)

for _i in range(TRAINING_ITERATIONS):
    optimizer.zero_grad()
    output = model.forward(x1)
    loss = -mll(output, full_train_y)

    loss.backward()
    optimizer.step()

model.eval()
likelihood.eval()

train_y1 = full_train_y[:, 0]
train_y2 = full_train_y[:, 1]
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
    torch.set_flush_denormal(True)
    xi = 0.9
    e = True
    counter2 = 0

    test_x = torch.from_numpy((dirichlet5D(500)) / 32).to(dtype=torch.float32)
    for i in range(int(test_x.shape[0] / 5)):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            ind = i * 5
            if i == 0:
                test_xt = test_x[0:5, :]
                mean, var = model.predict(test_xt)
                mean_t = mean
                var_t = var
            else:
                test_xt = test_x[ind : ind + 5, :]
                mean, var = model.predict(test_xt)
                mean_t = torch.cat((mean_t, mean), 0)
                var_t = torch.cat((var_t, var), 0)

    test_xt = test_x[ind + 5 :, :]
    if test_xt.numel() != 0:
        mean, var = model.predict(test_xt)
        mean_t = torch.cat((mean_t, mean), 0)
        var_t = torch.cat((var_t, var), 0)

    task = 0
    mean_t1 = mean_t[:, 0]
    mean_t2 = mean_t[:, 1]
    mean_t3 = mean_t[:, 2]
    var_t1 = var_t[:, 0].sqrt()
    var_t2 = var_t[:, 1].sqrt()
    var_t3 = var_t[:, 1].sqrt()
    eh_mean = np.concatenate(
        (np.array([mean_t1.numpy()]).T, np.array([mean_t2.numpy()]).T), axis=1
    )
    eh_std = np.concatenate(
        (np.array([var_t1.numpy()]).T, np.array([var_t2.numpy()]).T), axis=1
    )

    dgp_mean1_lst.append(mean_t1.numpy())
    dgp_mean2_lst.append(mean_t2.numpy())
    dgp_mean3_lst.append(mean_t3.numpy())
    dgp_std1_lst.append(var_t1.numpy())
    dgp_std2_lst.append(var_t2.numpy())
    dgp_std3_lst.append(var_t3.numpy())

    n_jobs = multiprocessing.cpu_count()

    def calc(ii):
        return EHVI(eh_mean[ii], eh_std[ii], goal, ref, y_pareto_truth)

    ehvi = Parallel(n_jobs)(delayed(calc)([jj]) for jj in range(eh_mean.shape[0]))
    ehvi = np.array(ehvi)

    x_star = np.argmax(ehvi)

    new_x = test_x.detach()[x_star]
    new_y = lookup_in_h5_file(new_x.unsqueeze(0), TEC_FILENAME, "tec")
    new_y2 = lookup_in_h5_file(new_x.unsqueeze(0), MOR_FILENAME, "bulkmodul_eq")
    new_y3 = lookup_in_h5_file(new_x.unsqueeze(0), MOR_FILENAME, "volume_eq")

    data_x = np.concatenate((x1.detach().numpy(), np.array([new_x.numpy()])), axis=0)
    data_y = np.concatenate((full_train_y[:, 0].detach().numpy(), new_y.numpy()), axis=0)

    x1_t = torch.tensor(data_x)
    train_y2_t = torch.tensor(data_y)
    data_y = np.concatenate((full_train_y[:, 1].detach().numpy(), new_y2.numpy()), axis=0)

    train_y3_t = torch.tensor(data_y)
    data_y = np.concatenate((full_train_y[:, 2].detach().numpy(), new_y3.numpy()), axis=0)

    train_y4_t = torch.tensor(data_y)

    y1_t = torch.stack(
        [
            train_y2_t,
            train_y3_t,
            train_y4_t,
        ],
        -1,
    )

    train_y1 = y1_t[:, 0]
    train_y2 = y1_t[:, 1]
    train_y3 = y1_t[:, 2]

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
    model = MultitaskIsoDeepGP(n_input_dims, n_output_dims, num_tasks).to(
        dtype=torch.float32
    )
    likelihod = model.likelihood

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.9)
    mll = DeepApproximateMLL(VariationalELBO(likelihood, model, num_data=y1.size(0)))

    for _i in range(TRAINING_ITERATIONS):
        optimizer.zero_grad()
        output = model(x1_t)
        loss = -mll(output, y1_t)

        loss.backward()
        optimizer.step()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = DeepApproximateMLL(
        VariationalELBO(likelihood, model, num_data=full_train_y.size(0))
    )

    for _i in range(TRAINING_ITERATIONS):
        optimizer.zero_grad()
        output = model(x1_t)
        loss = -mll(output, y1_t)

        loss.backward()
        optimizer.step()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = DeepApproximateMLL(
        VariationalELBO(likelihood, model, num_data=full_train_y.size(0))
    )

    for _i in range(TRAINING_ITERATIONS):
        optimizer.zero_grad()
        output = model(x1_t)
        loss = -mll(output, y1_t)

        loss.backward()
        optimizer.step()
        e = False

    x1 = deepcopy(x1_t)
    full_train_y = deepcopy(y1_t)
    train_y2 = deepcopy(train_y2_t)
    train_y3 = deepcopy(train_y3_t)
    train_y4 = deepcopy(train_y4_t)
    model.eval()
    likelihood.eval()

# %%
with h5py.File(file_out + ".h5", "w") as f:
    f.create_dataset("hv_truth", data=hv_truth)

    # Training data
    train_data = f.create_group("train")
    train_data.create_dataset("x1", data=x1)  # Note: x2 and x3 are same as x1
    train_data.create_dataset("y1", data=train_y2)
    train_data.create_dataset("y2", data=train_y3)
    train_data.create_dataset("y3", data=train_y4)

    # DGP predictions
    predictions = f.create_group("predictions")
    predictions.create_dataset("mean1", data=dgp_mean1_lst)
    predictions.create_dataset("mean2", data=dgp_mean2_lst)
    predictions.create_dataset("mean3", data=dgp_mean3_lst)
    predictions.create_dataset("std1", data=dgp_std1_lst)
    predictions.create_dataset("std2", data=dgp_std2_lst)
    predictions.create_dataset("std3", data=dgp_std3_lst)
