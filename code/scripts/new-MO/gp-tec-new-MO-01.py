import multiprocessing
import os

import gpytorch
import numpy as np
import torch
from joblib import Parallel, delayed

from dgp_bo import DATA_DIR
from dgp_bo.dirichlet import dirlet5D
from dgp_bo.multiobjective import EHVI, HV_Calc, Pareto_finder
from dgp_bo.utils import bmft, cft

SMOKE_TEST = "CI" in os.environ

file_out = os.path.basename(__file__)[:-3]
filename_global = os.path.join(DATA_DIR, "5space-mor.h5")
filename_global2 = os.path.join(DATA_DIR, "5space-md16-tec-new.h5")


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_constraint=gpytorch.constraints.Interval(0.3, 0.8)
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


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
opt_imp = []
debug_lst = []
debug_lst2 = []
debug_lst3 = []

hv_total_lst = []
# for b in range(20):
#     x1 = torch.from_numpy(normalize(torch.rand((2, 5)), axis=1, norm='l1'))#normalize(torch.rand((2, 5)), axis=1, norm='l1')
#     print("*****************************",b,"*******************************")
x1 = torch.from_numpy((dirlet5D(2, 5)) / 32)  # [np.random.randint(0,550,size=3),:]
# x2=torch.from_numpy((dirlet5D(2,5))/32)#[np.random.randint(0,550,size=3),:]


x2 = x1
y1 = bmft(x1)
y2 = cft(x1)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_constraint=gpytorch.constraints.Interval(0.2, 0.8)
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# initialize likelihood and model
likelihood1 = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.Interval(0.001, 10)
)
likelihood2 = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.Interval(0.001, 10)
)
model1 = ExactGPModel(x1, y1, likelihood1)
model2 = ExactGPModel(x2, y2, likelihood2)


# this is for running the notebook in our testing framework

SMOKE_TEST = "CI" in os.environ
training_iter = 2 if SMOKE_TEST else 500


# Find optimal model hyperparameters
model1.train()
likelihood1.train()

# Use the adam optimizer
optimizer1 = torch.optim.Adam(
    model1.parameters(), lr=0.1
)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll1 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood1, model1)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer1.zero_grad()
    # Output from model
    output1 = model1(x1)
    # Calc loss and backprop gradients
    loss1 = -mll1(output1, y1)
    loss1.backward()
    #     print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
    #         i + 1, training_iter, loss1.item(),
    #         model1.covar_module.base_kernel.lengthscale.item(),
    #         model1.likelihood.noise.item()
    #     ))
    optimizer1.step()

model1.eval()
likelihood1.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #     test_x = lhs(N_dim,30)
    #     test_x=torch.from_numpy(normalize(test_x, axis=1, norm='l1'))
    test_x1 = torch.from_numpy((dirlet5D(500, 5)) / 32)
    observed_pred1 = likelihood(model1(test_x1))


model2.train()
likelihood2.train()

# Use the adam optimizer
optimizer2 = torch.optim.Adam(
    model2.parameters(), lr=0.1
)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood2, model2)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer2.zero_grad()
    # Output from model
    output2 = model2(x2)
    # Calc loss and backprop gradients
    loss2 = -mll2(output2, y2)
    loss2.backward()
    #     print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
    #         i + 1, training_iter, loss2.item(),
    #         model2.covar_module.base_kernel.lengthscale.item(),
    #         model2.likelihood.noise.item()
    #     ))
    optimizer2.step()

model2.eval()
likelihood2.eval()
test_x1 = torch.from_numpy((dirlet5D(500, 5)) / 32)
# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #     test_x = lhs(N_dim,30)
    #     test_x=torch.from_numpy(normalize(test_x, axis=1, norm='l1'))

    observed_pred1 = likelihood1(model1(test_x1))

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #     test_x = lhs(N_dim,30)
    #     test_x=torch.from_numpy(normalize(test_x, axis=1, norm='l1'))
    # test_x1=torch.from_numpy((dirlet5D(500,5))/32)
    observed_pred2 = likelihood2(model2(test_x1))


#     x1=torch.from_numpy((dirlet5D(2,5))/32)[:3,:]

#     # b1=bmft(x1)[0]
#     # b2=cft(x2)[0]
#     # b3=c2ft(x3)[0]


#     y1 = bmft(x1)#/bmft(x1)[0]


mean_tp1 = observed_pred1.mean.detach()
mean_tp2 = observed_pred2.mean.detach()
yp = np.concatenate((y1.reshape(y1.shape[0], 1), y2.reshape(y2.shape[0], 1)), axis=1)
yp_query = yp.reshape(y1.shape[0], 2)

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

#     test_x = lhs(N_dim,550)
#     test_x=torch.from_numpy(normalize(test_x, axis=1, norm='l1'))

test_x = torch.from_numpy((dirlet5D(550, 5)) / 32)
test_x2 = test_x
test_x3 = test_x
test_x4 = test_x
for k in range(700):
    # test_x2 = torch.linspace(0, 1, 51)
    torch.cuda.empty_cache()
    torch.set_flush_denormal(True)
    xi = 0.3

    test_x = torch.from_numpy((dirlet5D(550, 5)) / 32)
    model1.eval()
    likelihood1.eval()

    for i in range(int(test_x.shape[0] / 5)):
        #             print(i)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            ind = i * 5
            if i == 0:
                test_xt = test_x[0:5, :]
                # print(test_xt,"******************************")
                # test_i_task1 = torch.full((test_xt.shape[0],1), dtype=torch.long, fill_value=0)
                observed_pred_y1 = likelihood1(model1(test_xt.float()))

                lower1, upper1 = observed_pred_y1.confidence_region()
                mean1 = observed_pred_y1.mean.detach()

                #                     print(lower)
                #                     print(upper)
                #                     print(mean)
                #                     print(var_t)
                mean_t1 = mean1
                var_t1 = (upper1.detach() - mean1) / 2
            else:
                test_xt = test_x[ind : ind + 5, :]
                #                     test_i_task1 = torch.full((test_xt.shape[0],1), dtype=torch.long, fill_value=0)
                observed_pred_y1 = likelihood1(model1(test_xt.float()))

                lower1, upper1 = observed_pred_y1.confidence_region()
                mean1 = observed_pred_y1.mean.detach()
                var1 = (upper1.detach() - mean1) / 2
                mean_t1 = torch.cat((mean_t1, mean1), 0)
                var_t1 = torch.cat((var_t1, var1), 0)
    test_xt = test_x[ind + 5 :, :]
    #         print(test_xt.float())
    if test_xt.numel() != 0:
        observed_pred_y1 = likelihood1(model1(test_xt.float()))

        lower1, upper1 = observed_pred_y1.confidence_region()
        mean1 = observed_pred_y1.mean.detach()
        var1 = (upper1.detach() - mean1) / 2
        mean_t1 = torch.cat((mean_t1, mean1), 0)
        var_t1 = torch.cat((var_t1, var1), 0)

    print(k)

    model2.eval()
    likelihood2.eval()

    for i in range(int(test_x.shape[0] / 5)):
        #             print(i)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            ind = i * 5
            if i == 0:
                test_xt = test_x[0:5, :]

                # test_i_task1 = torch.full((test_xt.shape[0],1), dtype=torch.long, fill_value=0)
                observed_pred_y2 = likelihood2(model2(test_xt.float()))

                lower2, upper2 = observed_pred_y2.confidence_region()
                mean2 = observed_pred_y2.mean.detach()

                #                     print(lower)
                #                     print(upper)
                #                     print(mean)
                #                     print(var_t)
                mean_t2 = mean2
                var_t2 = (upper2.detach() - mean2) / 2
            else:
                test_xt = test_x[ind : ind + 5, :]
                #                     test_i_task1 = torch.full((test_xt.shape[0],1), dtype=torch.long, fill_value=0)
                observed_pred_y2 = likelihood2(model2(test_xt.float()))

                lower2, upper2 = observed_pred_y2.confidence_region()
                mean2 = observed_pred_y2.mean.detach()
                var2 = (upper2.detach() - mean2) / 2
                mean_t2 = torch.cat((mean_t2, mean2), 0)
                var_t2 = torch.cat((var_t2, var2), 0)
    test_xt = test_x[ind + 5 :, :]
    #         print(test_xt.float())
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

    #     from sklearn.metrics import mutual_info_score #as mis
    #     from scipy.stats import entropy
    # #     import numpy as np
    #     #def pmis():

    #     c=0
    #     pmis=0
    #     ie1=0
    #     ie2=0
    #     kle=0
    #     cc=0
    #     for i in mean_t1.numpy():
    #         v1 = np.random.normal(mean_t1.numpy()[c], var_t1.numpy()[c], 100)
    # #         v1=(v1-mean_t1.numpy()[c])/var_t1.numpy()[c]
    #         v2 = np.random.normal(mean_t2.numpy()[c], var_t2.numpy()[c], 100)
    # #         v2=(v2-mean_t2.numpy()[c])/var_t2.numpy()[c]
    #         ie1=ie1+entropy(v1)
    #         ie2=ie2+entropy(v2)
    #         kle=kle+entropy(v1, qk=v2)
    #         pmis=pmis+mutual_info_score(v1,v2)
    #         cc=cc+np.corrcoef(v1,v2)
    #         c=c+1
    #     print(pmis,"&&&&&&&&&&&&&&&&&&")
    #     from scipy.stats import entropy
    # #     print("\nIndividual Entropy\n")
    # #     print(entropy(values1))
    # #     print(entropy(values2))
    # #     print(entropy(values3))

    # #     print("\nPairwise Kullback Leibler divergence\n")
    # #     print(entropy(values1, qk=values2))
    # #     print(entropy(values1, qk=values3))
    # #     print(entropy(values2, qk=values3))

    #     debug_lst.append(ie1)
    # #     debug_lst2.append(np.sum(var_t1.numpy()-var_t2.numpy()))
    #     debug_lst2.append(cc)
    #     debug_lst3.append(kle)
    #     mi_tb_lst.append(pmis)
    gp_mean1_lst.append(mean_t1.numpy())
    gp_mean2_lst.append(mean_t2.numpy())
    gp_std1_lst.append(var_t1.numpy())
    gp_std2_lst.append(var_t2.numpy())
    #     print(mean_t1.numpy(),"##########MEAN##########")
    #     print(mean_t2.numpy(),"##########MEAN##########")
    # y_pareto_truth,ind=eng.Pareto_finder_py(matlab.double(data_y),nargout=2)
    # y_pareto_truth=np.asarray(y_pareto_truth)

    # ind=np.asarray(ind)
    # ind=ind.astype(int)
    # ind=1
    # x_pareto_truth=data_x[ind]

    # hv_t=eng.HV(matlab.double(y_pareto_truth),nargout=1)
    # hv_truth = np.asarray(hv_t).reshape(1,1)

    n_jobs = multiprocessing.cpu_count()

    def calc(ii):
        return EHVI(eh_mean[ii], eh_std[ii], goal, ref, y_pareto_truth)

    ehvi = Parallel(n_jobs)(delayed(calc)([jj]) for jj in range(eh_mean.shape[0]))
    ehvi = np.array(ehvi)

    x_star = np.argmax(ehvi)

    #     kld_mv = np.log(std2/std1) + (std1**2 + (mean1 - mean2)**2)/(2*std2**2) - 0.5
    #         max_val, x_star, EI = expected_improvement(np.max(y1.numpy()), xi, mean_t.cpu().numpy(), var_t.cpu().numpy())
    #         print(EI,"#########################EI###################")
    new_x = test_x.detach()[x_star]

    new_y1 = bmft(new_x.unsqueeze(0))
    print(new_y1)
    data_x = np.concatenate((x1.numpy(), new_x.unsqueeze(0).numpy()), axis=0)
    #         try:
    # print(new_y1.numpy())
    # print(y1)
    data_y = np.concatenate((y1, new_y1.numpy()), axis=0)
    #         except:

    x1 = torch.tensor(data_x)
    y1 = torch.tensor(data_y)

    new_y2 = cft(new_x.unsqueeze(0))
    data_x = np.concatenate((x2.numpy(), new_x.unsqueeze(0).numpy()), axis=0)
    #         try:
    # print(new_y2.numpy())
    # print(y2)
    data_y = np.concatenate((y2, new_y2.numpy()), axis=0)
    #         except:

    x2 = torch.tensor(data_x)
    y2 = torch.tensor(data_y)

    #     test_x2=test_x2[test_x2!=new_x.item()]
    #     test_x2=test_x2.unsqueeze(1)

    yp = np.concatenate((y1.reshape(y1.shape[0], 1), y2.reshape(y1.shape[0], 1)), axis=1)
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
    print(hv_t, "hv_t")

    model1 = ExactGPModel(x1, y1, likelihood1)
    model2 = ExactGPModel(x2, y2, likelihood2)

    # this is for running the notebook in our testing framework
    import os

    SMOKE_TEST = "CI" in os.environ
    training_iter = 2 if SMOKE_TEST else 500

    # Find optimal model hyperparameters
    model1.train()
    likelihood1.train()

    # Use the adam optimizer
    optimizer1 = torch.optim.Adam(
        model1.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll1 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood1, model1)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer1.zero_grad()
        # Output from model
        output1 = model1(x1)
        # Calc loss and backprop gradients
        loss1 = -mll1(output1, y1)
        loss1.backward()
        #         print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        #             i + 1, training_iter, loss1.item(),
        #             model1.covar_module.base_kernel.lengthscale.item(),
        #             model1.likelihood.noise.item()
        #         ))
        optimizer1.step()

    model1.eval()
    likelihood1.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #     test_x = lhs(N_dim,30)
        #     test_x=torch.from_numpy(normalize(test_x, axis=1, norm='l1'))
        test_x1 = torch.from_numpy((dirlet5D(500, 5)) / 32)
        observed_pred1 = likelihood(model1(test_x1))

    model2.train()
    likelihood2.train()

    # Use the adam optimizer
    optimizer2 = torch.optim.Adam(
        model2.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood2, model2)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer2.zero_grad()
        # Output from model
        output2 = model2(x2)
        # Calc loss and backprop gradients
        loss2 = -mll2(output2, y2)
        loss2.backward()
        #         print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        #             i + 1, training_iter, loss2.item(),
        #             model2.covar_module.base_kernel.lengthscale.item(),
        #             model2.likelihood.noise.item()
        #         ))
        optimizer2.step()

    model2.eval()
    likelihood2.eval()


import pickle

with open(file_out + "-hv.pl", "wb") as handle:
    pickle.dump(hv_truth, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out + "-queryx1.pl", "wb") as handle:
    pickle.dump(x1, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out + "-queryx2.pl", "wb") as handle:
    pickle.dump(x2, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(file_out + "-queryy1.pl", "wb") as handle:
    pickle.dump(y1, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out + "-queryy2.pl", "wb") as handle:
    pickle.dump(y2, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(file_out + "-mean1.pl", "wb") as handle:
    pickle.dump(gp_mean1_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out + "-mean2.pl", "wb") as handle:
    pickle.dump(gp_mean2_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(file_out + "-std1.pl", "wb") as handle:
    pickle.dump(gp_std1_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out + "-std2.pl", "wb") as handle:
    pickle.dump(gp_std2_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
