# %%
import multiprocessing
import os

import gpytorch
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)
from joblib import Parallel, delayed

from dgp_bo import DATA_DIR
from dgp_bo.acquisitions import upper_conf_bound
from dgp_bo.dirichlet import dirlet5D
from dgp_bo.multiobjective import EHVI, HV_Calc, Pareto_finder
from dgp_bo.utils import bmft, c2ft, cft

# %%
SMOKE_TEST = "CI" in os.environ

file_out = os.path.basename(__file__)[:-3]
filename_global = os.path.join(DATA_DIR, "5space-mor.h5")
filename_global2 = os.path.join(DATA_DIR, "5space-md16-tec-new.h5")


class DGPLastLayer3(gpytorch.models.ApproximateGP):
    def __init__(self, num_inducing=16, linear_mean=True):
        num_latents = 10
        inducing_points = torch.randn(10, num_inducing, 3)
        torch.Size([10])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=torch.Size([num_latents])
        )

        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=3,
            num_latents=10,
            latent_dim=-1,
        )

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([10]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([10])),
            batch_shape=torch.Size([10]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, linear_mean=True):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = ConstantMean() if linear_mean else LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape,
            ard_num_dims=None,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class MultitaskDeepGP(DeepGP):
    def __init__(self, train_x_shape):
        hidden_layer = DGPHiddenLayer(input_dims=5, output_dims=3, linear_mean=True)
        last_layer = DGPLastLayer3(linear_mean=True)

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            num_tasks=3, noise_constraint=gpytorch.constraints.Interval(0.001, 10)
        )

    def forward(self, inputs, task_indices):
        task_indices1 = (
            torch.from_numpy(np.ones(inputs.shape[0], dtype=int) * 1).long()
            * task_indices
        )
        hidden_rep1 = self.hidden_layer(inputs)
        return self.last_layer(
            torch.distributions.Normal(
                loc=hidden_rep1.mean, scale=hidden_rep1.variance.sqrt()
            ).rsample(),
            task_indices=task_indices1,
        )
        #         print(output)

    def predict(self, test_x, task_indices):
        with torch.no_grad():
            preds = model.likelihood(model(test_x, task_indices=task_indices))

        return preds.mean, preds.variance


dgp_max_lst = []
dgp_full_lst = []
dgp_y_lst = []
dgp_x_lst = []
dgp_test_lst = []
dgp_pred_lst = []
dgp_std_lst = []
dgp_EI_lst = []
dgp_query_lst = []
dgp_mean1_lst = []
dgp_mean2_lst = []
dgp_mean3_lst = []
dgp_std1_lst = []
dgp_std2_lst = []
dgp_std3_lst = []

hv_total_lst = []
ref = np.array([[0, 0]])
goal = np.array([[1, 1]])
opt_imp = []


train_x1 = torch.from_numpy((dirlet5D(2, 5)) / 32)
train_x2 = train_x1
train_x3 = train_x1


train_y1 = bmft(train_x1, filename_global2)
train_y2 = cft(train_x2, filename_global)
train_y3 = c2ft(train_x3, filename_global)

train_i_task1 = torch.full((train_x1.shape[0], 1), dtype=torch.long, fill_value=0)
train_i_task2 = torch.full((train_x2.shape[0], 1), dtype=torch.long, fill_value=1)
train_i_task3 = torch.full((train_x3.shape[0], 1), dtype=torch.long, fill_value=2)

full_train_i = torch.cat([train_i_task1, train_i_task2, train_i_task3])
full_train_x = torch.cat([train_x1, train_x2, train_x3])
full_train_y = torch.cat([train_y1, train_y2, train_y3])

model = MultitaskDeepGP(train_x1.shape)
model = model
likelihood = model.likelihood
likelihood = likelihood


SMOKE_TEST = "CI" in os.environ
num_epochs = 1 if SMOKE_TEST else 500
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = DeepApproximateMLL(
    VariationalELBO(model.likelihood, model, num_data=full_train_y.size(0))
)

num_epochs = 1 if SMOKE_TEST else 500
# epochs_iter = tqdm_notebook(range(num_epochs), desc="Epoch")
epochs_iter = range(num_epochs)
for i in epochs_iter:
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

N_obj = 2
data_yp = yp_query


y_pareto_truth, ind = Pareto_finder(data_yp, goal)
hv_truth = (HV_Calc(goal, ref, y_pareto_truth)).reshape(1, 1)
train_y1t = torch.tensor(train_y1t)
train_y2t = torch.tensor(train_y2t)

for k in range(2):
    xi = 0.3

    torch.cuda.empty_cache()
    torch.set_flush_denormal(True)

    print("#######################", k, "######################")
    test_x = torch.from_numpy((dirlet5D(500, 5)) / 32)
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

    if (k % 3) == 0:
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
        new_y = bmft(new_x.unsqueeze(0), filename_global2)
        new_y2 = cft(new_x.unsqueeze(0), filename_global)
        new_y3 = c2ft(new_x.unsqueeze(0), filename_global)

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
        print(yp_query)
        N_obj = 2
        data_yp = yp

        y_pareto_truth, ind = Pareto_finder(data_yp, goal)
        hv_t = (HV_Calc(goal, ref, y_pareto_truth)).reshape(1, 1)
        hv_truth = np.concatenate((hv_truth, hv_t.reshape(1, 1)))

    if (k % 3) != 0:
        task = 0
        kt = 0
        max_val2, x_star2, UCB2 = upper_conf_bound(
            kt, mean_t2.cpu().numpy(), var_t2.cpu().numpy()
        )
        max_val3, x_star3, UCB3 = upper_conf_bound(
            kt, mean_t3.cpu().numpy(), var_t3.cpu().numpy()
        )

        new_x2 = test_x.detach()[x_star2]
        new_x3 = test_x.detach()[x_star3]

        new_y2 = cft(new_x2.unsqueeze(0))
        new_y3 = c2ft(new_x3.unsqueeze(0))

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

    model = MultitaskDeepGP(train_x1.shape)
    likelihood = model.likelihood

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = DeepApproximateMLL(
        VariationalELBO(likelihood, model, num_data=train_y1.size(0))
    )
    num_epochs = 500 if (counter < 250) is True else 250
    epochs_iter = range(num_epochs)
    for _i in epochs_iter:
        optimizer.zero_grad()
        output = model(full_train_x.float(), task_indices=full_train_i.squeeze(-1))
        loss = -mll(output, full_train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    counter = counter + 1

# %%
import pickle

with open(file_out + "-hv.pl", "wb") as handle:
    pickle.dump(hv_truth, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out + "-queryx1.pl", "wb") as handle:
    pickle.dump(train_x1, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out + "-queryx2.pl", "wb") as handle:
    pickle.dump(train_x2, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(file_out + "-queryx3.pl", "wb") as handle:
    pickle.dump(train_x3, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(file_out + "-queryy1.pl", "wb") as handle:
    pickle.dump(train_y1, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out + "-queryy2.pl", "wb") as handle:
    pickle.dump(train_y2, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out + "-queryy3.pl", "wb") as handle:
    pickle.dump(train_y3, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(file_out + "-mean1.pl", "wb") as handle:
    pickle.dump(dgp_mean1_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out + "-mean2.pl", "wb") as handle:
    pickle.dump(dgp_mean2_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(file_out + "-mean3.pl", "wb") as handle:
    pickle.dump(dgp_mean3_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(file_out + "-std1.pl", "wb") as handle:
    pickle.dump(dgp_std1_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out + "-std2.pl", "wb") as handle:
    pickle.dump(dgp_std2_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out + "-std3.pl", "wb") as handle:
    pickle.dump(dgp_std3_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
