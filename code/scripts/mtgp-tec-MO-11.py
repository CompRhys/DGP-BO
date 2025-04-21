import pandas as pd
import sys

index = sys.argv[1]
file_out = sys.argv[0][:-3] + sys.argv[1]

import torch
import gpytorch

# from matplotlib import pyplot as plt
import numpy as np
from pyDOE import *
from dgp_bo.multiobjective import Pareto_finder
import os

# from mymod import call_Curtin_Model , call_kkr
import multiprocessing
from joblib import Parallel, delayed

# import matlab.engine
# eng = matlab.engine.start_matlab()
from dgp_bo.multiobjective import EHVI, HV_Calc


# from matplotlib import pyplot as plt
from pyDOE import *



# import matplotlib.pyplot as plt
# from acquisitionFuncdebug2 import expected_improvement


def pos(x):
    if x >= 0:
        return x
    elif x < 0:
        return 0




def dirlet5D(N_samp, dim):
    from scipy.stats import dirichlet

    out = []
    n = 5
    size = N_samp
    alpha = np.ones(n)
    samples = dirichlet.rvs(size=size, alpha=alpha)
    # print(samples)
    samples2 = np.asarray(
        [
            np.asarray(
                [
                    round(i * 32),
                    round(j * 32),
                    round(k * 32),
                    round(l * 32),
                    pos(
                        32
                        - round(i * 32)
                        - round(j * 32)
                        - round(k * 32)
                        - round(l * 32)
                    ),
                ]
            )
            for i, j, k, l, m in samples
        ]
    )
    for i, j, k, l, m in samples2:
        if i / 32 + j / 32 + k / 32 + l / 32 + m / 32 == 1:
            out.append(np.asarray([i, j, k, l, m]))
        else:
            out.append(dirlet5D(1, 5)[0])
    #             print("********************exception**********")

    return np.asarray(out)


filename_global = "5space-mor.h5"
filename_global2 = "5space-md16-tec-new.h5"


def bmft(x, bmf="tec", filename=filename_global2):
    xt = x.cpu().numpy()
    c11 = []
    conc = np.asarray([np.asarray([i, j, k, l, m]) for i, j, k, l, m in xt])
    for x, y, z, u, v in conc:
        df1 = pd.read_hdf(filename)

        c = df1.loc[
            (df1["Fe"] == x)
            & (df1["Cr"] == y)
            & (df1["Ni"] == z)
            & (df1["Co"] == u)
            & (df1["Cu"] == v)
        ][bmf].values[0]
        # print(c,"****")
        # print(df1.loc[(df1["conc_Fe"]==x)]["C11"].values[0])
        c11.append(c)

    return torch.from_numpy(np.asarray(c11))  # .cuda()

    return torch.from_numpy(y)


def cft(x, bmf="bulkmodul_eq", filename=filename_global):
    xt = x.cpu().numpy()

    c11 = []
    conc = np.asarray([np.asarray([i, j, k, l, m]) for i, j, k, l, m in xt])
    for x, y, z, u, v in conc:
        df1 = pd.read_hdf(filename)

        c = df1.loc[
            (df1["Fe"] == x)
            & (df1["Cr"] == y)
            & (df1["Ni"] == z)
            & (df1["Co"] == u)
            & (df1["Cu"] == v)
        ][bmf].values[0]
        # print(c,"****")
        # print(df1.loc[(df1["conc_Fe"]==x)]["C11"].values[0])
        c11.append(c)

    return torch.from_numpy(np.array(c11))  # .cuda()


def c2ft(x, bmf="volume_eq", filename=filename_global):
    xt = x.cpu().numpy()
    c11 = []
    conc = np.asarray([np.asarray([i, j, k, l, m]) for i, j, k, l, m in xt])
    for x, y, z, u, v in conc:
        df1 = pd.read_hdf(filename)

        c = df1.loc[
            (df1["Fe"] == x)
            & (df1["Cr"] == y)
            & (df1["Ni"] == z)
            & (df1["Co"] == u)
            & (df1["Cu"] == v)
        ][bmf].values[0]
        # print(c,"****")
        # print(df1.loc[(df1["conc_Fe"]==x)]["C11"].values[0])
        c11.append(c)

    return torch.from_numpy(np.asarray(c11))  # .cuda()


from gpytorch import settings

mtgp_max_lst = []
mtgp_y_lst = []
mtgp_x_lst = []
mtgp_test_lst = []
mtgp_pred_lst = []
mtgp_std_lst = []
mtgp_EI_lst = []
mtgp_query_lst = []
hv_total_lst = []

mtgp_mean1_lst = []
mtgp_mean2_lst = []
mtgp_mean3_lst = []
mtgp_std1_lst = []
mtgp_std2_lst = []
mtgp_std3_lst = []

ref = np.array([[0, 0]])
goal = np.array([[1, 1]])
opt_imp = []
settings.debug.off()

# train_x1 = torch.from_numpy(normalize(torch.rand((1, 5)), axis=1, norm='l1'))#normalize(torch.rand((2, 5)), axis=1, norm='l1')
train_x1 = torch.from_numpy(
    (dirlet5D(2, 5)) / 32
)  # [np.random.randint(0,550,size=3),:]
train_x2 = train_x1
train_x3 = train_x1


train_y1 = bmft(train_x1)  # + torch.randn(train_x1.size()) * 0.2
train_y2 = cft(train_x2)  # + torch.randn(train_x2.size()) * 0.2
train_y3 = c2ft(train_x3)


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.MaternKernel(
            lengthscale_constraint=gpytorch.constraints.Interval(0.2, 0.8)
        )  # lengthscale_constraint=gpytorch.constraints.Interval(0.01, 0.1)
        # MaternKernel
        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)
        self.task_covar_module = gpytorch.kernels.IndexKernel(
            num_tasks=3, rank=3
        )  # gpytorch.constraints.Interval(-1, 1)),var_constraint=gpytorch.constraints.GreaterThan(-0.5)

    def forward(self, x, i):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)


likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.Interval(0.001, 10)
)  # noise_constraint=gpytorch.constraints.Interval(1, 5)

train_i_task1 = torch.full((train_x1.shape[0], 1), dtype=torch.long, fill_value=0)
train_i_task2 = torch.full((train_x2.shape[0], 1), dtype=torch.long, fill_value=1)
train_i_task3 = torch.full((train_x3.shape[0], 1), dtype=torch.long, fill_value=2)
# train_i_task4 = torch.full((train_x4.shape[0],1), dtype=torch.long, fill_value=3)

full_train_x = torch.cat([train_x1, train_x2, train_x3])
full_train_i = torch.cat([train_i_task1, train_i_task2, train_i_task3])
full_train_y = torch.cat([train_y1, train_y2, train_y3])

# Here we have two iterms that we're passing in as train_inputs
model = MultitaskGPModel((full_train_x, full_train_i), full_train_y, likelihood)

# this is for running the notebook in our testing framework

smoke_test = "CI" in os.environ
training_iterations = 2 if smoke_test else 500


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.1
)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(full_train_x, full_train_i)
    loss = -mll(output, full_train_y)
    loss.backward()
    # print('Iter %d/50 - Loss: %.3f' % (i + 1, loss.item()))
    optimizer.step()

# Set into eval mode
model.eval()
likelihood.eval()

yp = np.concatenate(
    (train_y1.reshape(train_y1.shape[0], 1), train_y2.reshape(train_y2.shape[0], 1)),
    axis=1,
)
yp_query = yp.reshape(train_y1.shape[0], 2)

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


test_x = torch.from_numpy((dirlet5D(500, 5)) / 32)
for k in range(500):
    print("**************", k, "************************")
    torch.cuda.empty_cache()
    torch.set_flush_denormal(True)
    # test_x2 = torch.linspace(0, 1, 51)
    xi = 0.9
    # test_x = torch.linspace(0, 1, 51)

    #         test_x = lhs(N_dim,100)
    #         test_x=torch.from_numpy(normalize(test_x, axis=1, norm='l1'))
    test_x = torch.from_numpy((dirlet5D(500, 5)) / 32)
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
    mean_t1 = observed_pred_y1.mean.detach()  # .numpy()
    var_t1 = torch.tensor(std1)
    upper2, lower2 = observed_pred_y2.confidence_region()
    std2 = (lower2.numpy() - observed_pred_y2.mean.detach().numpy()) / 2
    mean_t2 = observed_pred_y2.mean.detach()  # .numpy()
    var_t2 = torch.tensor(std2)

    upper3, lower3 = observed_pred_y3.confidence_region()
    std3 = (lower3.numpy() - observed_pred_y3.mean.detach().numpy()) / 2
    mean_t3 = observed_pred_y3.mean.detach()  # .numpy()
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
        e = EHVI(eh_mean[ii], eh_std[ii], goal, ref, y_pareto_truth)
        return e

    ehvi = Parallel(n_jobs)(delayed(calc)([jj]) for jj in range(eh_mean.shape[0]))
    ehvi = np.array(ehvi)

    x_star = np.argmax(ehvi)
    #         for i in range(int(test_x.shape[0]/5)):

    #             with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #                 ind=i*5
    #                 if i ==0:
    #                     test_xt=test_x[0:5,:]
    #                     observed_pred_y1 = likelihood(model(test_xt, test_i_task1[0:5,:]))
    #                     mean=observed_pred_y1.mean.detach()
    #                     upper,lower=observed_pred_y1.confidence_region()
    #                     var=(lower-mean)/2
    #                     mean_t=mean
    #                     var_t=var
    #                 else:
    #                     test_xt=test_x[ind-5:ind,:]
    #                     observed_pred_y1 = likelihood(model(test_xt, test_i_task1[ind-5:ind,:]))
    #                     mean=observed_pred_y1.mean.detach()
    #                     upper,lower=observed_pred_y1.confidence_region()
    #                     var=(lower-mean)/2
    #                     mean_t=torch.cat((mean_t,mean),0)
    #                     var_t=torch.cat((var_t,var),0)

    #         test_i_task1 = torch.full((test_x.shape[0],1), dtype=torch.long, fill_value=0)
    #     test_i_task2 = torch.full((test_x2.shape[0],1), dtype=torch.long, fill_value=1)
    #     test_i_task3 = torch.full((test_x2.shape[0],1), dtype=torch.long, fill_value=2)

    #     # Make predictions - one task at a time
    #     # We control the task we cae about using the indices

    #     # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
    #     # See https://arxiv.org/abs/1803.06058
    #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #         observed_pred_y1 = likelihood(model(test_x2, test_i_task1))
    #         observed_pred_y2 = likelihood(model(test_x2, test_i_task2))
    #         observed_pred_y3 = likelihood(model(test_x2, test_i_task3))

    #     print(k)
    #     print(np.array(observed_pred_y1.confidence_region()).shape)

    #     upper,lower=observed_pred_y1.confidence_region()
    #     std=lower.numpy()-observed_pred_y1.mean.detach().numpy()
    # print(var_t)

    #     for bb in range(test_x2.numpy().shape[0]):
    #             train_x_temp=torch.cat((train_x2,test_x2[bb].unsqueeze(-2)),axis=0)
    #             train_y_temp=torch.cat((train_y2,samp[bb].unsqueeze(-2)),axis=0)
    #             model_temp=deepcopy(model)
    #             model_temp.train()
    #             likelihood.train()
    #             model_temp.set_train_data(train_x_temp,train_y_temp,strict=False)
    #             model_temp.eval()
    #             likelihood.eval()
    #             with torch.no_grad():
    #                 predictions = model_temp(x_test)
    #                 mean_t = predictions.mean
    #                 std_t = predictions.stddev

    #     kld_mv = np.log(std2/std1) + (std1**2 + (mean1 - mean2)**2)/(2*std2**2) - 0.5
    #         max_val, x_star, EI = expected_improvement(np.max(train_y1.numpy()), xi, observed_pred_y1.mean.detach().numpy(), std)

    new_x = test_x.detach()[x_star]
    # new_x=torch.rand(100)[x_star]

    #         if b==2:
    #             mtgp_pred_lst.append(observed_pred_y1.mean.detach().numpy())
    #             mtgp_std_lst.append(std)
    #             mtgp_EI_lst.append(EI)
    #             #gp_query_lst()
    #     #         new_x=test_x2.detach()[x_star]
    #             mtgp_query_lst.append(new_x)
    #             mtgp_test_lst.append(test_x)
    test_x = torch.cat((test_x[:x_star], test_x[x_star + 1 :]))
    #     train_y1 = train_x1 ** (2)
    #     train_y2 = (train_x2 **(2))*(-1)
    new_y = cft(new_x.unsqueeze(0))
    new_y2 = c2ft(new_x.unsqueeze(0))

    #     new_y= torch.cos(new_x * (2 * math.pi))# + torch.randn(new_x.size()) * 0.2

    data_x = np.concatenate((train_x2.numpy(), new_x.unsqueeze(0).numpy()), axis=0)
    data_y = np.concatenate((train_y2, new_y.numpy()), axis=0)
    train_x2 = torch.tensor(data_x)
    train_y2 = torch.tensor(data_y)
    data_x = np.concatenate((train_x3.numpy(), new_x.unsqueeze(0).numpy()), axis=0)
    data_y = np.concatenate((train_y3, new_y2.numpy()), axis=0)
    train_x3 = torch.tensor(data_x)
    train_y3 = torch.tensor(data_y)
    #     data_x=np.concatenate((train_x4.numpy(),new_x.numpy().reshape(1)),axis=0)
    #     data_y=np.concatenate((train_y4,new_y3.numpy().reshape(1)),axis=0)
    #     train_x4=torch.tensor(data_x)
    #     train_y4=torch.tensor(data_y)

    #     if (k%10)==0:
    # new_y1= torch.sin(new_x * (2 * math.pi))
    new_y1 = bmft(new_x.unsqueeze(0))
    data_x = np.concatenate((train_x1.numpy(), new_x.unsqueeze(0).numpy()), axis=0)
    data_y = np.concatenate((train_y1, new_y1.numpy()), axis=0)
    train_x1 = torch.tensor(data_x)
    train_y1 = torch.tensor(data_y)

    #     test_x2=test_x2[test_x2!=new_x.item()]

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
    # print(yp_query)
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

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Interval(0.001, 10)
    )
    model = MultitaskGPModel((full_train_x, full_train_i), full_train_y, likelihood)

    import os

    smoke_test = "CI" in os.environ
    training_iterations = 2 if smoke_test else 500

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    #         model = MultitaskGPModel((full_train_x, full_train_i), full_train_y, likelihood)
    #         for i in range(int(full_train_x.shape[0]/5)):

    #             ind=i*5
    #             if i ==0:
    #                 train_xt=full_train_x[0:5,:]
    #                 train_it=full_train_i[0:5,:]
    #                 train_yt=full_train_y[0:5]
    #             else:

    #                 train_xt=full_train_x[ind-5:ind,:]
    #                 train_it=full_train_i[ind-5:ind,:]
    #                 train_yt=full_train_y[ind-5:ind]

    #             import os
    #             smoke_test = ('CI' in os.environ)
    #             training_iterations = 2 if smoke_test else 50

    #             # Find optimal model hyperparameters
    #             model.train()
    #             likelihood.train()

    #             # Use the adam optimizer
    #             optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    #             # "Loss" for GPs - the marginal log likelihood
    #             mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    #             print(train_xt,train_it,train_yt,"&&&&&&&&&&&&&&&&")
    #             for i in range(training_iterations):
    #                 optimizer.zero_grad()
    #                 output = model(train_xt, train_it)
    #                 loss = -mll(output, train_yt)
    #                 loss.backward()
    #                 #print('Iter %d/50 - Loss: %.3f' % (i + 1, loss.item()))
    #                 optimizer.step()

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(full_train_x, full_train_i)
        loss = -mll(output, full_train_y)
        loss.backward()
        # print('Iter %d/500 - Loss: %.3f' % (i + 1, loss.item()))

        #             print(model.task_covar_module)
        optimizer.step()
    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Initialize plots

    # Test points every 0.02 in [0,1]
    #     test_x = lhs(N_dim,550)
    #     test_x=torch.from_numpy(normalize(test_x, axis=1, norm='l1'))
    #     test_i_task1 = torch.full((test_x.shape[0],1), dtype=torch.long, fill_value=0)
    #     test_i_task2 = torch.full((test_x.shape[0],1), dtype=torch.long, fill_value=1)
    #     test_i_task3 = torch.full((test_x.shape[0],1), dtype=torch.long, fill_value=2)
    #     test_i_task4 = torch.full((test_x.shape[0],1), dtype=torch.long, fill_value=3)

    #     # Make predictions - one task at a time
    #     # We control the task we cae about using the indices

    #     # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
    #     # See https://arxiv.org/abs/1803.06058
    #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #         observed_pred_y1 = likelihood(model(test_x, test_i_task1))
    #         observed_pred_y2 = likelihood(model(test_x, test_i_task2))
    #         observed_pred_y3 = likelihood(model(test_x, test_i_task3))

#         if b==2:
#             mtgp_y_lst.append(train_y1.detach().numpy())
#             mtgp_x_lst.append(train_x1.detach().numpy())
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
    pickle.dump(mtgp_mean1_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out + "-mean2.pl", "wb") as handle:
    pickle.dump(mtgp_mean2_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(file_out + "-mean3.pl", "wb") as handle:
    pickle.dump(mtgp_mean3_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(file_out + "-std1.pl", "wb") as handle:
    pickle.dump(mtgp_std1_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out + "-std2.pl", "wb") as handle:
    pickle.dump(mtgp_std2_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out + "-std3.pl", "wb") as handle:
    pickle.dump(mtgp_std3_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
