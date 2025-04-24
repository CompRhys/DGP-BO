# %%
import multiprocessing
import os

import gpytorch
import h5py
import numpy as np
import torch
import tqdm.auto as tqdm
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from joblib import Parallel, delayed

from dgp_bo import DATA_DIR
from dgp_bo.acquisitions import upper_conf_bound
from dgp_bo.dirichlet import dirichlet5D
from dgp_bo.mtdgp import MultitaskHetDeepGP
from dgp_bo.multiobjective import EHVI, HV_Calc, Pareto_finder
from dgp_bo.utils import lookup_in_h5_file

# %%
SMOKE_TEST = True
TRAINING_ITERATIONS = 2 if SMOKE_TEST else 500
BO_ITERATIONS = 10 if SMOKE_TEST else 500
CHEAP_EXPENSIVE_RATIO = 3

file_out = os.path.basename(__file__)[:-3]
MOR_FILENAME = os.path.join(DATA_DIR, "5space-mor.h5")
TEC_FILENAME = os.path.join(DATA_DIR, "5space-md16-tec-new.h5")

n_output_dims = 3
n_tasks = 3

dgp_mean1_lst = []
dgp_mean2_lst = []
dgp_mean3_lst = []
dgp_std1_lst = []
dgp_std2_lst = []
dgp_std3_lst = []

ref = np.array([[180, 0]])
goal = np.array([[0, 1]])

train_x1 = torch.from_numpy((dirichlet5D(2)) / 32)
train_x2 = train_x1
train_x3 = train_x1

train_y1 = lookup_in_h5_file(train_x1, TEC_FILENAME, "tec")
train_y2 = lookup_in_h5_file(train_x2, MOR_FILENAME, "bulkmodul_eq")
train_y3 = lookup_in_h5_file(train_x3, MOR_FILENAME, "volume_eq")

train_i_task1 = torch.full((train_x1.shape[0], 1), dtype=torch.long, fill_value=0)
train_i_task2 = torch.full((train_x2.shape[0], 1), dtype=torch.long, fill_value=1)
train_i_task3 = torch.full((train_x3.shape[0], 1), dtype=torch.long, fill_value=2)

full_train_i = torch.cat([train_i_task1, train_i_task2, train_i_task3])
full_train_x = torch.cat([train_x1, train_x2, train_x3])
full_train_y = torch.cat([train_y1, train_y2, train_y3])

model = MultitaskHetDeepGP(train_x1.shape, n_output_dims, n_tasks)
likelihood = model.likelihood

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = DeepApproximateMLL(
    VariationalELBO(model.likelihood, model, num_data=full_train_y.size(0))
)

for _i in range(TRAINING_ITERATIONS):
    optimizer.zero_grad()
    output = model(full_train_x.float(), task_indices=full_train_i.squeeze(-1))
    loss = -mll(output, full_train_y)
    loss.backward()
    optimizer.step()

counter = 0
train_y1t = train_y1.cpu().numpy()
train_y2t = train_y2.cpu().numpy()
yp = np.concatenate(
    (
        train_y1t.reshape(train_y1t.shape[0], 1),
        train_y2t.reshape(train_y2t.shape[0], 1),
    ),
    axis=1,
)
yp_query = yp.reshape(train_y1t.shape[0], 2)

data_yp = yp_query
y_pareto_truth, ind = Pareto_finder(data_yp, goal)
hv_truth = (HV_Calc(goal, ref, y_pareto_truth)).reshape(1, 1)
train_y1t = torch.tensor(train_y1t)
train_y2t = torch.tensor(train_y2t)


# %%
for k in tqdm.tqdm(range(BO_ITERATIONS)):
    xi = 0.3
    torch.cuda.empty_cache()
    torch.set_flush_denormal(True)

    test_x = torch.from_numpy((dirichlet5D(500)) / 32)
    for i in range(int(test_x.shape[0] / 5)):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            ind = i * 5
            if i == 0:
                test_xt = test_x[0:5, :]
                test_i_task1 = torch.full(
                    (test_xt.shape[0], 1), dtype=torch.long, fill_value=0
                )
                test_i_task2 = torch.full(
                    (test_xt.shape[0], 1), dtype=torch.long, fill_value=1
                )
                test_i_task3 = torch.full(
                    (test_xt.shape[0], 1), dtype=torch.long, fill_value=2
                )

                observed_pred_y1 = model.likelihood(
                    model(test_xt.float(), task_indices=test_i_task1.squeeze(1))
                )
                observed_pred_y2 = model.likelihood(
                    model(test_xt.float(), task_indices=test_i_task2.squeeze(1))
                )
                observed_pred_y3 = model.likelihood(
                    model(test_xt.float(), task_indices=test_i_task3.squeeze(1))
                )

                lower1, upper1 = observed_pred_y1.confidence_region()
                mean1 = observed_pred_y1.mean.detach()
                mean_t1 = mean1
                var_t1 = (upper1.detach() - mean1) / 2

                lower2, upper2 = observed_pred_y2.confidence_region()
                mean2 = observed_pred_y2.mean.detach()
                mean_t2 = mean2
                var_t2 = (upper2.detach() - mean2) / 2

                lower3, upper3 = observed_pred_y3.confidence_region()
                mean3 = observed_pred_y3.mean.detach()
                mean_t3 = mean3
                var_t3 = (upper3.detach() - mean3) / 2

            else:
                test_xt = test_x[ind : ind + 5, :]
                test_i_task1 = torch.full(
                    (test_xt.shape[0], 1), dtype=torch.long, fill_value=0
                )
                test_i_task2 = torch.full(
                    (test_xt.shape[0], 1), dtype=torch.long, fill_value=1
                )
                test_i_task3 = torch.full(
                    (test_xt.shape[0], 1), dtype=torch.long, fill_value=2
                )

                observed_pred_y1 = model.likelihood(
                    model(test_xt.float(), task_indices=test_i_task1.squeeze(1))
                )
                observed_pred_y2 = model.likelihood(
                    model(test_xt.float(), task_indices=test_i_task2.squeeze(1))
                )
                observed_pred_y3 = model.likelihood(
                    model(test_xt.float(), task_indices=test_i_task3.squeeze(1))
                )

                lower1, upper1 = observed_pred_y1.confidence_region()
                mean1 = observed_pred_y1.mean.detach()
                var1 = (upper1.detach() - mean1) / 2
                mean_t1 = torch.cat((mean_t1, mean1), 0)
                var_t1 = torch.cat((var_t1, var1), 0)

                lower2, upper2 = observed_pred_y2.confidence_region()
                mean2 = observed_pred_y2.mean.detach()
                var2 = (upper2.detach() - mean2) / 2
                mean_t2 = torch.cat((mean_t2, mean2), 0)
                var_t2 = torch.cat((var_t2, var2), 0)

                lower3, upper3 = observed_pred_y3.confidence_region()
                mean3 = observed_pred_y3.mean.detach()
                var3 = (upper3.detach() - mean3) / 2
                mean_t3 = torch.cat((mean_t3, mean3), 0)
                var_t3 = torch.cat((var_t3, var3), 0)

    test_xt = test_x[ind + 5 :, :]
    if test_xt.numel() != 0:
        test_i_task1 = torch.full((test_xt.shape[0], 1), dtype=torch.long, fill_value=0)
        test_i_task2 = torch.full((test_xt.shape[0], 1), dtype=torch.long, fill_value=1)
        test_i_task3 = torch.full((test_xt.shape[0], 1), dtype=torch.long, fill_value=2)

        observed_pred_y1 = model.likelihood(
            model(test_xt.float(), task_indices=test_i_task1.squeeze(1))
        )
        observed_pred_y2 = model.likelihood(
            model(test_xt.float(), task_indices=test_i_task2.squeeze(1))
        )
        observed_pred_y3 = model.likelihood(
            model(test_xt.float(), task_indices=test_i_task3.squeeze(1))
        )

        lower1, upper1 = observed_pred_y1.confidence_region()
        mean1 = observed_pred_y1.mean.detach()
        var1 = (upper1.detach() - mean1) / 2
        mean_t1 = torch.cat((mean_t1, mean1), 0)
        var_t1 = torch.cat((var_t1, var1), 0)

        lower2, upper2 = observed_pred_y2.confidence_region()
        mean2 = observed_pred_y2.mean.detach()
        var2 = (upper2.detach() - mean2) / 2
        mean_t2 = torch.cat((mean_t2, mean2), 0)
        var_t2 = torch.cat((var_t2, var2), 0)

        lower3, upper3 = observed_pred_y3.confidence_region()
        mean3 = observed_pred_y3.mean.detach()
        var3 = (upper3.detach() - mean3) / 2
        mean_t3 = torch.cat((mean_t3, mean3), 0)
        var_t3 = torch.cat((var_t3, var3), 0)

    dgp_mean1_lst.append(mean_t1.cpu().numpy())
    dgp_mean2_lst.append(mean_t2.cpu().numpy())
    dgp_mean3_lst.append(mean_t3.cpu().numpy())
    dgp_std1_lst.append(var_t1.cpu().numpy())
    dgp_std2_lst.append(var_t2.cpu().numpy())
    dgp_std3_lst.append(var_t3.cpu().numpy())

    if (k % CHEAP_EXPENSIVE_RATIO) == 0:
        task = 0
        eh_mean = np.concatenate(
            (np.array([mean_t1.cpu().numpy()]).T, np.array([mean_t2.cpu().numpy()]).T),
            axis=1,
        )
        eh_std = np.concatenate(
            (np.array([var_t1.cpu().numpy()]).T, np.array([var_t2.cpu().numpy()]).T),
            axis=1,
        )

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

        data_x = np.concatenate(
            (train_x1.detach().cpu().numpy(), np.array([new_x.cpu().numpy()])), axis=0
        )
        data_y = np.concatenate(
            (train_y1.detach().cpu().numpy(), new_y.cpu().numpy().reshape(1)), axis=0
        )
        train_x1 = torch.tensor(data_x)
        train_y1 = torch.tensor(data_y)
        data_x = np.concatenate(
            (train_x2.detach().cpu().numpy(), np.array([new_x.cpu().numpy()])), axis=0
        )
        data_y = np.concatenate(
            (train_y2.detach().cpu().numpy(), new_y2.cpu().numpy().reshape(1)), axis=0
        )
        train_x2 = torch.tensor(data_x)
        train_y2 = torch.tensor(data_y)
        data_x = np.concatenate(
            (train_x3.detach().cpu().numpy(), np.array([new_x.cpu().numpy()])), axis=0
        )
        data_y = np.concatenate(
            (train_y3.detach().cpu().numpy(), new_y3.cpu().numpy().reshape(1)), axis=0
        )
        train_x3 = torch.tensor(data_x)
        train_y3 = torch.tensor(data_y)

        data_y = np.concatenate(
            (train_y1t.detach().cpu().numpy(), new_y.cpu().numpy().reshape(1)), axis=0
        )

        train_y1t = torch.tensor(data_y)

        data_y = np.concatenate(
            (train_y2t.detach().cpu().numpy(), new_y2.cpu().numpy().reshape(1)), axis=0
        )
        train_y2t = torch.tensor(data_y)

        yp = np.concatenate(
            (
                train_y1t.cpu().numpy().reshape(train_y1t.cpu().numpy().shape[0], 1),
                train_y2t.cpu().numpy().reshape(train_y2t.cpu().numpy().shape[0], 1),
            ),
            axis=1,
        )
        yp_query = yp.reshape(yp.shape[0], 2)
        data_yp = yp

        y_pareto_truth, ind = Pareto_finder(data_yp, goal)
        hv_t = (HV_Calc(goal, ref, y_pareto_truth)).reshape(1, 1)
        hv_truth = np.concatenate((hv_truth, hv_t.reshape(1, 1)))

    if (k % CHEAP_EXPENSIVE_RATIO) != 0:
        task = 0
        kt = 0
        max_val2, x_star2, _ucb2 = upper_conf_bound(
            kt, mean_t2.cpu().numpy(), var_t2.cpu().numpy()
        )
        max_val3, x_star3, _ucb3 = upper_conf_bound(
            kt, mean_t3.cpu().numpy(), var_t3.cpu().numpy()
        )

        new_x2 = test_x.detach()[x_star2]
        new_x3 = test_x.detach()[x_star3]
        new_y2 = lookup_in_h5_file(new_x2.unsqueeze(0), MOR_FILENAME, "bulkmodul_eq")
        new_y3 = lookup_in_h5_file(new_x3.unsqueeze(0), MOR_FILENAME, "volume_eq")
        data_x = np.concatenate(
            (train_x2.detach().cpu().numpy(), np.array([new_x2.cpu().numpy()])), axis=0
        )
        data_y = np.concatenate(
            (train_y2.detach().cpu().numpy(), new_y2.cpu().numpy().reshape(1)), axis=0
        )
        train_x2 = torch.tensor(data_x)
        train_y2 = torch.tensor(data_y)
        data_x = np.concatenate(
            (train_x3.detach().cpu().numpy(), np.array([new_x3.cpu().numpy()])), axis=0
        )
        data_y = np.concatenate(
            (train_y3.detach().cpu().numpy(), new_y3.cpu().numpy().reshape(1)), axis=0
        )
        train_x3 = torch.tensor(data_x)
        train_y3 = torch.tensor(data_y)

    train_i_task1 = torch.full((train_x1.shape[0], 1), dtype=torch.long, fill_value=0)
    train_i_task2 = torch.full((train_x2.shape[0], 1), dtype=torch.long, fill_value=1)
    train_i_task3 = torch.full((train_x3.shape[0], 1), dtype=torch.long, fill_value=2)

    full_train_i = torch.cat([train_i_task1, train_i_task2, train_i_task3])
    full_train_x = torch.cat([train_x1, train_x2, train_x3])
    full_train_y = torch.cat([train_y1, train_y2, train_y3])

    model = MultitaskHetDeepGP(train_x1.shape, n_output_dims, n_tasks)
    likelihood = model.likelihood

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = DeepApproximateMLL(
        VariationalELBO(likelihood, model, num_data=train_y1.size(0))
    )
    for _i in range(TRAINING_ITERATIONS):
        optimizer.zero_grad()
        output = model(full_train_x.float(), task_indices=full_train_i.squeeze(-1))
        loss = -mll(output, full_train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    counter = counter + 1

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

    # DGP predictions
    predictions = f.create_group("predictions")
    predictions.create_dataset("mean1", data=dgp_mean1_lst)
    predictions.create_dataset("mean2", data=dgp_mean2_lst)
    predictions.create_dataset("mean3", data=dgp_mean3_lst)
    predictions.create_dataset("std1", data=dgp_std1_lst)
    predictions.create_dataset("std2", data=dgp_std2_lst)
    predictions.create_dataset("std3", data=dgp_std3_lst)
