import multiprocessing
import os
import sys

import gpytorch
import numpy as np
import pandas as pd
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

from dgp_bo.dirichlet import dirlet5D
from dgp_bo.multiobjective import EHVI, HV_Calc, Pareto_finder

SMOKE_TEST = "CI" in os.environ
index = sys.argv[1]
file_out = sys.argv[0][:-3] + sys.argv[1]


df = pd.read_hdf("5space-mor.h5")
df.head()

filename_global = "5space-mor.h5"
filename_global2 = "5space-md16-tec-new.h5"


def bmft(x, bmf="tec", filename=filename_global2):
    xt = x.cpu().numpy()
    c11 = []
    conc = np.asarray([np.asarray([i, j, k, l, m]) for i, j, k, l, m in xt])
    for x, y, z, u, v in conc:
        df1 = pd.read_hdf(filename)

        #         print(x,y,z,u,v,"###################################333")
        c = df1.loc[
            (df1["Fe"] == x)
            & (df1["Cr"] == y)
            & (df1["Ni"] == z)
            & (df1["Co"] == u)
            & (df1["Cu"] == v)
        ][bmf].values[0]
        print(c, "****")
        # print(df1.loc[(df1["conc_Fe"]==x)]["C11"].values[0])
        c11.append(c)

    return torch.from_numpy(np.asarray(c11)).cuda()

    # return torch.from_numpy(y)


def cft(x, bmf="bulkmodul_eq", filename=filename_global):
    xt = x.cpu().numpy()

    c11 = []
    conc = np.asarray([np.asarray([i, j, k, l, m]) for i, j, k, l, m in xt])
    for x, y, z, u, v in conc:
        df1 = pd.read_hdf(filename)

        #         print(x,y,z,u,v,"###################################333")
        c = df1.loc[
            (df1["Fe"] == x)
            & (df1["Cr"] == y)
            & (df1["Ni"] == z)
            & (df1["Co"] == u)
            & (df1["Cu"] == v)
        ][bmf].values[0]
        #         print(c*10**7,"****")
        print(c, "****")
        # print(df1.loc[(df1["conc_Fe"]==x)]["C11"].values[0])
        c11.append(c)

    return torch.from_numpy(np.array(c11)).cuda()


def c2ft(x, bmf="volume_eq", filename=filename_global):
    xt = x.cpu().numpy()
    c11 = []
    conc = np.asarray([np.asarray([i, j, k, l, m]) for i, j, k, l, m in xt])
    for x, y, z, u, v in conc:
        df1 = pd.read_hdf(filename)

        #         print(x,y,z,u,v,"###################################333")
        c = df1.loc[
            (df1["Fe"] == x)
            & (df1["Cr"] == y)
            & (df1["Ni"] == z)
            & (df1["Co"] == u)
            & (df1["Cu"] == v)
        ][bmf].values[0]
        #         print(c*10**7,"****")
        print(c, "****")
        # print(df1.loc[(df1["conc_Fe"]==x)]["C11"].values[0])
        c11.append(c)

    return torch.from_numpy(np.asarray(c11)).cuda()


class DGPLastLayer3(gpytorch.models.ApproximateGP):
    def __init__(self, num_inducing=16, linear_mean=True):
        num_latents = 10
        inducing_points = torch.randn(10, num_inducing, 3)
        torch.Size([10])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=torch.Size([num_latents])
        )
        #         variational_strategy = VariationalStrategy(
        #             self,
        #             inducing_points,
        #             variational_distribution,
        #             learn_inducing_locations=True
        #         )

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
        #         self.covar_module = ScaleKernel(
        #             MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
        #             batch_shape=batch_shape, ard_num_dims=None
        #         )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                batch_shape=torch.Size([10])
            ),  # ,lengthscale_constraint=gpytorch.constraints.Interval(0.2, 0.8)
            batch_shape=torch.Size([10]),
        )

    #     def forward(self, x):
    #         mean_x = self.mean_module(x)
    #         covar_x = self.covar_module(x)
    #         return MultivariateNormal(mean_x, covar_x)

    def forward(self, x):
        #         print('in DGP', x.shape)
        #         x = torch.distributions.Normal(loc=x.mean, scale=x.variance.sqrt()).rsample()
        mean_x = self.mean_module(x)
        #         print('mx', mean_x.shape)
        covar_x = self.covar_module(x)
        #         print('cx', covar_x.shape)
        return MultivariateNormal(mean_x, covar_x)
        #         print('y', y.shape)


#     def __call__(self, inputs, are_samples=False, **kwargs):
#         deterministic_inputs = not are_samples
#         if isinstance(inputs, MultitaskMultivariateNormal):
#             inputs = torch.distributions.Normal(loc=inputs.mean, scale=inputs.variance.sqrt()).rsample()#.unsqueeze(-3)
#             deterministic_inputs = False
#         print(inputs.shape)
#         output = ApproximateGP.__call__(self, inputs, **kwargs)
#         print(inputs)
#         print(inputs[0,:])
#         print(kwargs)
#         output = ApproximateGP.__call__(self, inputs[0,:], **kwargs)
#         return output
#             print("DeepGPLayer-input",inputs.shape)
#         if self.output_dims is not None:
#         inputs = inputs.unsqueeze(-3)
#         print("unsqueezed",inputs.shape)
# #         print("expand args",*inputs.shape[:-3],self.output_dims,*inputs.shape[-2:])
#         print("unexapnded inputs",inputs.shape,"###################")
#         inputs = inputs.expand(*inputs.shape[:-3], 1, *inputs.shape[-2:])
#         print("exapnded inputs",inputs.shape,"#############")
#         # Now run samples through the GP
#         print(inputs.shape)
#         print(inputs)

#         print("output.batch_shape",output.batch_shape)
#         print("DeepGPLayer-output",output.shape)
#         if self.output_dims is not None:
#         mean = output.loc.transpose(-1, -2)
#         covar = BlockDiagLinearOperator(output.lazy_covariance_matrix, block_dim=-3)
#         output = MultitaskMultivariateNormal(mean, covar, interleaved=False)

#         print("MultitaskMultivariateNormal-output",output.mean.shape)
#         # Maybe expand inputs?
#         print("output.batch_shape",output.batch_shape)
# #         if deterministic_inputs:
# #             output = output.expand(torch.Size([settings.num_likelihood_samples.value()]) + output.batch_shape)
#         print("determ_output",output.shape)
# #         return output#.unsqueeze(-1)


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
            MaternKernel(
                nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims
            ),  # ,lengthscale_constraint=gpytorch.constraints.Interval(0.2, 0.8)
            batch_shape=batch_shape,
            ard_num_dims=None,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# full_train_x.unsqueeze(-1).shape

# train_x_shape


class MultitaskDeepGP(DeepGP):
    def __init__(self, train_x_shape):
        hidden_layer = DGPHiddenLayer(input_dims=5, output_dims=3, linear_mean=True)
        #         hidden_layer = DGPHiddenLayer(

        #             linear_mean=True
        #         )
        last_layer = DGPLastLayer3(linear_mean=True)
        #         last_layer = DGPLastLayer(
        #             input_dims=hidden_layer.output_dims,
        #             output_dims=num_tasks,
        #             linear_mean=False
        #         )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer

        # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            num_tasks=3, noise_constraint=gpytorch.constraints.Interval(0.001, 10)
        )  # ,noise_constraint=gpytorch.constraints.Interval(0.001, 10)

    def forward(self, inputs, task_indices):
        #         hidden_rep1 = self.hidden_layer(inputs,task_indices=torch.tensor(i).unsqueeze(-1))
        #         print(torch.tensor(i).unsqueeze(-1))
        #         print(hidden_rep1)
        #         output = self.last_layer(hidden_rep1,task_indices=torch.tensor(i).unsqueeze(-1))
        #         print(task_indices)
        # print(inputs)
        # print(torch.from_numpy(np.ones(inputs.shape[0],dtype=int)).long()*task_indices)
        task_indices1 = (
            torch.from_numpy(np.ones(inputs.shape[0], dtype=int) * 1).long().cuda()
            * task_indices
        )
        hidden_rep1 = self.hidden_layer(inputs)  # ,task_indices=task_indices1
        #         print(torch.distributions.Normal(loc=hidden_rep1.mean, scale=hidden_rep1.variance.sqrt()).rsample().unsqueeze(1),"**********hidden_rep1 output****************")
        # print(torch.from_numpy(np.ones(torch.distributions.Normal(loc=hidden_rep1.mean, scale=hidden_rep1.variance.sqrt()).rsample())).shape)

        #         print(task_indices1.shape,"###############indices shape####################")
        #         task_indices2=task_indices1.expand(3,-1).expand(10,-1,-1)      #task_indices1.expand(3,-1).unsqueeze(1).expand(10,-1,-1,-1)
        # output=self.last_layer(torch.distributions.Normal(loc=hidden_rep1.mean, scale=hidden_rep1.variance.sqrt()).rsample(),task_indices=task_indices2)
        #         output=self.last_layer(torch.distributions.Normal(loc=hidden_rep1.mean, scale=hidden_rep1.variance.sqrt()).rsample().unsqueeze(-1),task_indices=task_indices1)
        #         print(torch.distributions.Normal(loc=hidden_rep1.mean, scale=hidden_rep1.variance.sqrt()).rsample().shape,"sample shape")
        return self.last_layer(
            torch.distributions.Normal(
                loc=hidden_rep1.mean, scale=hidden_rep1.variance.sqrt()
            ).rsample(),
            task_indices=task_indices1.cuda(),
        )
        #         print(output)

    def predict(self, test_x, task_indices):
        with torch.no_grad():
            # The output of the model is a multitask MVN, where both the data points
            # and the tasks are jointly distributed
            # To compute the marginal predictive NLL of each data point,
            # we will call `to_data_independent_dist`,
            # which removes the data cross-covariance terms from the distribution.
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
ref = np.array([[180, 0]])
goal = np.array([[0, 1]])
opt_imp = []


from acquisitionFuncdebug2 import upper_conf_bound

train_x1 = torch.from_numpy(
    (dirlet5D(2, 5)) / 32
).cuda()  # [np.random.randint(0,557,size=3),:]
train_x2 = train_x1
train_x3 = train_x1


train_y1 = bmft(train_x1)  # /bmft(x1)[0]
train_y2 = cft(train_x2)  # /cft(x2)[0]
train_y3 = c2ft(train_x3)  # /c2ft(x3)[0]


# y1 = torch.stack([
#     y1,
#     y2,
#     y3,
# ], -1)

train_i_task1 = torch.full((train_x1.shape[0], 1), dtype=torch.long, fill_value=0)
train_i_task2 = torch.full((train_x2.shape[0], 1), dtype=torch.long, fill_value=1)
train_i_task3 = torch.full((train_x3.shape[0], 1), dtype=torch.long, fill_value=2)

full_train_i = torch.cat([train_i_task1, train_i_task2, train_i_task3])
full_train_x = torch.cat([train_x1, train_x2, train_x3])
full_train_y = torch.cat([train_y1, train_y2, train_y3])

model = MultitaskDeepGP(train_x1.shape)
model = model.cuda()
likelihood = model.likelihood
likelihood = likelihood.cuda()

import os

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
    output = model(full_train_x.float(), task_indices=full_train_i.squeeze(-1).cuda())
    loss = -mll(output, full_train_y)
    #     output = model(train_x)
    #     loss = -mll(output, train_y)
    #    epochs_iter.set_postfix(loss=loss.item())
    loss.backward()
    optimizer.step()


#     test_x=torch.from_numpy((dirlet5D(100,5))/32)
# test_x = lhs(N_dim,100)
# test_x=torch.from_numpy(normalize(test_x, axis=1, norm='l1'))
# test_x2 = test_x
# test_x3 = test_x
# test_x4 = test_x
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
# data_x=np.concatenate((data_x,x_query.reshape(1,N_dim)),axis=0)
data_yp = yp_query


# y_pareto_truth,ind=eng.Pareto_finder_py(matlab.double(data_y),nargout=2)
# y_pareto_truth=np.asarray(y_pareto_truth)
y_pareto_truth, ind = Pareto_finder(data_yp, goal)
# ind=np.asarray(ind)
# ind=ind.astype(int)
# ind=1
# x_pareto_truth=data_x[ind]
# hv_t=eng.HV(matlab.double(y_pareto_truth),nargout=1)
# hv_t=np.asarray(hv_t).reshape(1,1)
#     hv_t = (HV_Calc(goal,ref,y_pareto_truth)).reshape(1,1)
hv_truth = (HV_Calc(goal, ref, y_pareto_truth)).reshape(1, 1)
train_y1t = torch.tensor(train_y1t).cuda()
train_y2t = torch.tensor(train_y2t).cuda()

for k in range(1500):
    # test_x2 = torch.linspace(0, 1, 51)
    xi = 0.3
    # test_x = torch.linspace(0, 1, 51)

    torch.cuda.empty_cache()
    torch.set_flush_denormal(True)

    print("#######################", k, "######################")
    #     test_x = lhs(N_dim,500)
    #     test_x=torch.from_numpy(normalize(test_x, axis=1, norm='l1'))
    test_x = torch.from_numpy((dirlet5D(500, 5)) / 32)
    for i in range(int(test_x.shape[0] / 5)):
        #             print(i)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            ind = i * 5
            if i == 0:
                test_xt = test_x[0:5, :].cuda()
                test_i_task1 = torch.full(
                    (test_xt.shape[0], 1), dtype=torch.long, fill_value=0
                ).cuda()
                test_i_task2 = torch.full(
                    (test_xt.shape[0], 1), dtype=torch.long, fill_value=1
                ).cuda()
                test_i_task3 = torch.full(
                    (test_xt.shape[0], 1), dtype=torch.long, fill_value=2
                ).cuda()

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
                test_xt = test_x[ind : ind + 5, :].cuda()
                test_i_task1 = torch.full(
                    (test_xt.shape[0], 1), dtype=torch.long, fill_value=0
                ).cuda()
                test_i_task2 = torch.full(
                    (test_xt.shape[0], 1), dtype=torch.long, fill_value=1
                ).cuda()
                test_i_task3 = torch.full(
                    (test_xt.shape[0], 1), dtype=torch.long, fill_value=2
                ).cuda()

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

    test_xt = test_x[ind + 5 :, :].cuda()
    #         print(test_xt.float())
    if test_xt.numel() != 0:
        test_i_task1 = torch.full(
            (test_xt.shape[0], 1), dtype=torch.long, fill_value=0
        ).cuda()
        test_i_task2 = torch.full(
            (test_xt.shape[0], 1), dtype=torch.long, fill_value=1
        ).cuda()
        test_i_task3 = torch.full(
            (test_xt.shape[0], 1), dtype=torch.long, fill_value=2
        ).cuda()

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
        #     kld_mv = np.log(std2/std1) + (std1**2 + (mean1 - mean2)**2)/(2*std2**2) - 0.5
        #             max_val, x_star, EI = expected_improvement(np.max(train_y1.detach().cpu().numpy()), xi,mean_t1.cpu().numpy(), var_t1.cpu().numpy())#.sqrt()

        new_x = test_x.detach()[x_star]
        # new_x=torch.rand(100)[x_star]

        #     train_y1 = train_x1 ** (2)
        #     train_y2 = (train_x2 **(2))*(-1)
        new_y = bmft(new_x.unsqueeze(0))
        new_y2 = cft(new_x.unsqueeze(0))
        new_y3 = c2ft(new_x.unsqueeze(0))

        #             test_x = torch.cat((test_x[:x_star],test_x[x_star+1:]))

        #     new_y= torch.cos(new_x * (2 * math.pi))# + torch.randn(new_x.size()) * 0.2

        data_x = np.concatenate(
            (train_x1.detach().cpu().numpy(), np.array([new_x.cpu().numpy()])), axis=0
        )
        data_y = np.concatenate(
            (train_y1.detach().cpu().numpy(), new_y.cpu().numpy().reshape(1)), axis=0
        )
        train_x1 = torch.tensor(data_x).cuda()
        train_y1 = torch.tensor(data_y).cuda()
        data_x = np.concatenate(
            (train_x2.detach().cpu().numpy(), np.array([new_x.cpu().numpy()])), axis=0
        )
        data_y = np.concatenate(
            (train_y2.detach().cpu().numpy(), new_y2.cpu().numpy().reshape(1)), axis=0
        )
        train_x2 = torch.tensor(data_x).cuda()
        train_y2 = torch.tensor(data_y).cuda()
        data_x = np.concatenate(
            (train_x3.detach().cpu().numpy(), np.array([new_x.cpu().numpy()])), axis=0
        )
        data_y = np.concatenate(
            (train_y3.detach().cpu().numpy(), new_y3.cpu().numpy().reshape(1)), axis=0
        )
        train_x3 = torch.tensor(data_x).cuda()
        train_y3 = torch.tensor(data_y).cuda()

        data_y = np.concatenate(
            (train_y1t.detach().cpu().numpy(), new_y.cpu().numpy().reshape(1)), axis=0
        )

        train_y1t = torch.tensor(data_y).cuda()

        data_y = np.concatenate(
            (train_y2t.detach().cpu().numpy(), new_y2.cpu().numpy().reshape(1)), axis=0
        )
        train_y2t = torch.tensor(data_y).cuda()

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
        # data_x=np.concatenate((data_x,x_query.reshape(1,N_dim)),axis=0)
        data_yp = yp

        # y_pareto_truth,ind=eng.Pareto_finder_py(matlab.double(data_y),nargout=2)
        # y_pareto_truth=np.asarray(y_pareto_truth)
        y_pareto_truth, ind = Pareto_finder(data_yp, goal)
        # ind=np.asarray(ind)
        # ind=ind.astype(int)
        # ind=1
        # x_pareto_truth=data_x[ind]
        # hv_t=eng.HV(matlab.double(y_pareto_truth),nargout=1)
        # hv_t=np.asarray(hv_t).reshape(1,1)
        hv_t = (HV_Calc(goal, ref, y_pareto_truth)).reshape(1, 1)
        hv_truth = np.concatenate((hv_truth, hv_t.reshape(1, 1)))
        print(hv_t, "hv_t")

    # y1 = torch.stack([
    #     y1,
    #     y2,
    #     y3,
    # ], -1)

    if (k % 3) != 0:
        task = 0
        kt = 0
        #     kld_mv = np.log(std2/std1) + (std1**2 + (mean1 - mean2)**2)/(2*std2**2) - 0.5
        #         max_val, x_star, EI = expected_improvement(np.max(train_y1.detach().cpu().numpy()), xi,mean_t.cpu().numpy(), var_t.sqrt().cpu().numpy())
        max_val2, x_star2, UCB2 = upper_conf_bound(
            kt, mean_t2.cpu().numpy(), var_t2.cpu().numpy()
        )  # .sqrt()
        max_val3, x_star3, UCB3 = upper_conf_bound(
            kt, mean_t3.cpu().numpy(), var_t3.cpu().numpy()
        )  # .sqrt()

        new_x2 = test_x.detach()[x_star2]
        # new_x=torch.rand(100)[x_star]
        new_x3 = test_x.detach()[x_star3]

        #     train_y1 = train_x1 ** (2)
        #     train_y2 = (train_x2 **(2))*(-1)
        #         new_y=bmft(new_x.unsqueeze(0))
        new_y2 = cft(new_x2.unsqueeze(0))
        new_y3 = c2ft(new_x3.unsqueeze(0))

        #         test_x = torch.cat((test_x[:x_star],test_x[x_star+1:]))

        #     new_y= torch.cos(new_x * (2 * math.pi))# + torch.randn(new_x.size()) * 0.2

        #         data_x=np.concatenate((train_x1.detach().cpu().numpy(), np.array([new_x.cpu().numpy()])),axis=0)
        #         data_y=np.concatenate((train_y1.detach().cpu().numpy(),new_y.cpu().numpy().reshape(1)),axis=0)
        #         train_x1=torch.tensor(data_x).cuda()
        #         train_y1=torch.tensor(data_y).cuda()
        data_x = np.concatenate(
            (train_x2.detach().cpu().numpy(), np.array([new_x2.cpu().numpy()])), axis=0
        )
        data_y = np.concatenate(
            (train_y2.detach().cpu().numpy(), new_y2.cpu().numpy().reshape(1)), axis=0
        )
        train_x2 = torch.tensor(data_x).cuda()
        train_y2 = torch.tensor(data_y).cuda()
        data_x = np.concatenate(
            (train_x3.detach().cpu().numpy(), np.array([new_x3.cpu().numpy()])), axis=0
        )
        data_y = np.concatenate(
            (train_y3.detach().cpu().numpy(), new_y3.cpu().numpy().reshape(1)), axis=0
        )
        train_x3 = torch.tensor(data_x).cuda()
        train_y3 = torch.tensor(data_y).cuda()

    train_i_task1 = torch.full((train_x1.shape[0], 1), dtype=torch.long, fill_value=0)
    train_i_task2 = torch.full((train_x2.shape[0], 1), dtype=torch.long, fill_value=1)
    train_i_task3 = torch.full((train_x3.shape[0], 1), dtype=torch.long, fill_value=2)

    full_train_i = torch.cat([train_i_task1, train_i_task2, train_i_task3])
    full_train_x = torch.cat([train_x1, train_x2, train_x3])
    full_train_y = torch.cat([train_y1, train_y2, train_y3])

    #     if (k%10)==0:
    #         #new_y1= torch.sin(new_x * (2 * math.pi))
    #         new_y1=bmft(torch.Tensor([new_x]))
    #         data_x=np.concatenate((train_x1.numpy(),new_x.numpy().reshape(1)),axis=0)
    #         data_y=np.concatenate((train_y1,new_y1.numpy().reshape(1)),axis=0)
    #         train_x1=torch.tensor(data_x)
    #         train_y1=torch.tensor(data_y)

    #     test_x2=test_x2[test_x2!=new_x.item()]

    #     train_i_task1 = torch.full((train_x1.shape[0],1), dtype=torch.long, fill_value=0)
    #     train_i_task2 = torch.full((train_x2.shape[0],1), dtype=torch.long, fill_value=1)
    #     train_i_task3 = torch.full((train_x3.shape[0],1), dtype=torch.long, fill_value=2)
    #     train_i_task4 = torch.full((train_x4.shape[0],1), dtype=torch.long, fill_value=3)

    #     full_train_x = torch.cat([train_x1, train_x2,train_x3,train_x4])
    #     full_train_i = torch.cat([train_i_task1, train_i_task2,train_i_task3,train_i_task4])
    #     full_train_y = torch.cat([train_y1, train_y2,train_y3,train_y4])

    # model = MultitaskGPModel((full_train_x, full_train_i), full_train_y, likelihood)

    model = MultitaskDeepGP(train_x1.shape)
    model = model.cuda()
    likelihood = model.likelihood
    likelihood = likelihood.cuda()

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = DeepApproximateMLL(
        VariationalELBO(likelihood, model, num_data=train_y1.size(0))
    )
    num_epochs = 500 if (counter < 250) is True else 250
    #    epochs_iter = tqdm_notebook(range(num_epochs), desc="Epoch")
    epochs_iter = range(num_epochs)
    for i in epochs_iter:
        optimizer.zero_grad()
        output = model(full_train_x.float(), task_indices=full_train_i.squeeze(-1).cuda())
        loss = -mll(output, full_train_y)
        #        epochs_iter.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    counter = counter + 1
#     with torch.no_grad(), gpytorch.settings.fast_pred_var():
#         test_x = lhs(N_dim,100)
#         test_x=torch.from_numpy(normalize(test_x, axis=1, norm='l1'))
#         mean, var = model.predict(test_x.float())
#         lower = mean - 2 * var.sqrt()
#         upper = mean + 2 * var.sqrt()
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
