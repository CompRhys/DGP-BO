import pandas as pd
import sys
index=sys.argv[1]
file_out=sys.argv[0][:-3]+sys.argv[1]
import torch
import gpytorch
# from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from pyDOE import *
from copy import deepcopy
from multiobjective import Pareto_finder
import os
import shutil
from sklearn.preprocessing import normalize
from gpModel import gp_model
# from mymod import call_Curtin_Model , call_kkr
from multiprocessing import Pool
import multiprocessing
from joblib import Parallel, delayed
#import matlab.engine
#eng = matlab.engine.start_matlab()
import random
from MeanCov1 import MeanCov1 
from multiobjective import EHVI, Pareto_finder, HV_Calc

import torch
import gpytorch
# from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from pyDOE import *
from copy import deepcopy

import os
import shutil
from sklearn.preprocessing import normalize
from gpModel import gp_model

import math
import torch
import gpytorch
# import matplotlib.pyplot as plt
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import _HomoskedasticNoiseBase
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor

import gpytorch
import numpy as np
#from acquisitionFuncdebug2 import expected_improvement

import math
import torch
import gpytorch
# from matplotlib import pyplot as plt

def pos(x):
    if x>=0:
        return x
    elif x<0:
        return 0

import numpy as np
def dirlet5D(N_samp,dim):    
    from scipy.stats import dirichlet
    out=[]
    n = 5
    size = N_samp
    alpha = np.ones(n)
    samples = dirichlet.rvs(size=size, alpha=alpha)
    #print(samples)
    samples2=np.asarray([np.asarray([round(i*32),round(j*32),round(k*32),round(l*32),pos(32-round(i*32)-round(j*32)-round(k*32)-round(l*32))]) for i,j,k,l,m in samples ])
    for i,j,k,l,m in samples2:
        if i/32+j/32+k/32+l/32+m/32==1:
            out.append(np.asarray([i,j,k,l,m]))
        else:
            out.append((dirlet5D(1,5)[0]))
#             print("********************exception**********")
    
    
    return np.asarray(out)



filename_global="5space-md16-tec-new.h5"
filename_global2="5space-mor.h5"
def bmft(x,bmf='tec',filename=filename_global):
    
    xt=x.numpy()
    c11=[]
    conc=np.asarray([np.asarray([i,j,k,l,m]) for i,j,k,l,m in xt])
    for x,y,z,u,v in conc:
        
        df1=pd.read_hdf(filename) 
        
#         print(x,y,z,u,v,"###################################333")
        c=df1.loc[(df1["Fe"] == x) & (df1["Cr"] == y) & (df1["Ni"] == z)& (df1["Co"] == u)& (df1["Cu"] == v) ][bmf].values[0]
        
        #print(df1.loc[(df1["conc_Fe"]==x)]["C11"].values[0])
        c11.append(c)
     
    return torch.from_numpy(np.asarray(c11))
    
    
def cft(x,bmf='bulkmodul_eq',filename=filename_global2):
    xt=x.numpy()
    
    c11=[]
    conc=np.asarray([np.asarray([i,j,k,l,m]) for i,j,k,l,m in xt])
    for x,y,z,u,v in conc:
        
        df1=pd.read_hdf(filename) 
        
#         print(x,y,z,u,v,"###################################333")
        c=df1.loc[(df1["Fe"] == x) & (df1["Cr"] == y) & (df1["Ni"] == z)& (df1["Co"] == u)& (df1["Cu"] == v) ][bmf].values[0]
#         print(c*10**7,"****")
        
        #print(df1.loc[(df1["conc_Fe"]==x)]["C11"].values[0])
        c11.append(c)
     
    return torch.from_numpy(np.array(c11))
def c2ft(x,bmf='volume_eq',filename=filename_global2):
    xt=x.numpy()
    c11=[]
    conc=np.asarray([np.asarray([i,j,k,l,m]) for i,j,k,l,m in xt])
    for x,y,z,u,v in conc:
        
        df1=pd.read_hdf(filename) 
        
#         print(x,y,z,u,v,"###################################333")
        c=df1.loc[(df1["Fe"] == x) & (df1["Cr"] == y) & (df1["Ni"] == z)& (df1["Co"] == u)& (df1["Cu"] == v) ][bmf].values[0]
#         print(c*10**7,"****")
        
        #print(df1.loc[(df1["conc_Fe"]==x)]["C11"].values[0])
        c11.append(c)
     
    return torch.from_numpy(np.asarray(c11))

import os
import torch

import math
import gpytorch
from torch.nn import Linear
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, \
    LMCVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood
#from matplotlib import pyplot as plt

smoke_test = ('CI' in os.environ)
#%matplotlib inline

# Here's a simple standard layer

# Here's a simple standard layer

class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, linear_mean=True):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
            jitter_val=1e-3
        )

        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = ConstantMean() if linear_mean else LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims,lengthscale_constraint=gpytorch.constraints.Interval(0.2, 0.8)), #,lengthscale_constraint=gpytorch.constraints.Interval(0.3, 0.8)
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

num_tasks = 3
num_hidden_dgp_dims = 3
#num_hidden_dgp_dims2 = 3


class MultitaskDeepGP(DeepGP):
    def __init__(self, train_x_shape):
        hidden_layer = DGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dgp_dims,
            linear_mean=True
        )
        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=num_tasks,
            linear_mean=False
        )
#         last_layer2 = DGPHiddenLayer(
#             input_dims=last_layer.output_dims,
#             output_dims=num_tasks,
#             linear_mean=False
#         )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
#         self.last_layer2=last_layer2
        # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks,noise_constraint=gpytorch.constraints.Interval(0.001, 10)) #,noise_constraint=gpytorch.constraints.Interval(0.001, 10)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
#         output2 = self.last_layer2(output)
        return output

    def predict(self, test_x):
        with torch.no_grad():

            # The output of the model is a multitask MVN, where both the data points
            # and the tasks are jointly distributed
            # To compute the marginal predictive NLL of each data point,
            # we will call `to_data_independent_dist`,
            # which removes the data cross-covariance terms from the distribution.
            preds = model.likelihood(model(test_x)).to_data_independent_dist()

        return preds.mean.mean(0), preds.variance.mean(0)



dgp_max_lst=[]
dgp_y_lst=[]
dgp_x_lst=[]
dgp_test_lst=[]
dgp_pred_lst=[]
dgp_std_lst=[]
dgp_EI_lst=[]
dgp_query_lst=[]
dgp_max_tmp=[]
dgp_conc_lst=[]


dgp_mean1_lst=[]
dgp_mean2_lst=[]
dgp_mean3_lst=[]
dgp_std1_lst=[]
dgp_std2_lst=[]
dgp_std3_lst=[]
# dgp_max_tmp.append(np.arange(1,5))
hv_total_lst=[]
ref=np.array([[0,0]])
goal=np.array([[1,1]])
opt_imp=[]

#     x1 = torch.from_numpy(normalize(torch.rand((2, 5)), axis=1, norm='l1')).cuda()#normalize(torch.rand((2, 5)), axis=1, norm='l1')
x1=torch.from_numpy((dirlet5D(2,5))/32)#[np.random.randint(0,557,size=3),:]
x2 = x1
x3= x1






y1 = bmft(x1)#/bmft(x1)[0]
y2 = cft(x2)#/cft(x2)[0]
y3 = c2ft(x3)#/c2ft(x3)[0]

y1 = torch.stack([
    y1,
    y2,
    y3,
], -1)


# Here's a simple standard layer

# Here's a simple standard layer

class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=64, linear_mean=True):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )


#             variational_strategy = gpytorch.variational.BatchDecoupledVariationalStrategy(
#                  self, inducing_points, variational_distribution, learn_inducing_locations=True,
#                  mean_var_batch_dim=-1
#              )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
            jitter_val=1e-3
        )

#             grid_bounds=[(0, 100),(0, 100),(0, 100),(0, 100),(0, 100)]
#             grid_size=64
#             variational_strategy = gpytorch.variational.GridInterpolationVariationalStrategy(
#                     self, grid_size=grid_size, grid_bounds=[grid_bounds],
#                     variational_distribution=variational_distribution,
#                 )


        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = ConstantMean() if linear_mean else LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=None,lengthscale_constraint=gpytorch.constraints.Interval(0.2, 0.8)), #,lengthscale_constraint=gpytorch.constraints.Interval(0.3, 0.8)
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

num_tasks = 3
num_hidden_dgp_dims = 4
#num_hidden_dgp_dims2 = 3


class MultitaskDeepGP(DeepGP):
    def __init__(self, train_x_shape):
        hidden_layer = DGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dgp_dims,
            linear_mean=True
        )
        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=num_tasks,
            linear_mean=False
        )
#         last_layer2 = DGPHiddenLayer(
#             input_dims=last_layer.output_dims,
#             output_dims=num_tasks,
#             linear_mean=False
#         )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
#         self.last_layer2=last_layer2
        # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood

        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks,noise_constraint=gpytorch.constraints.Interval(0.001, 10)) #,noise_constraint=gpytorch.constraints.Interval(0.001, 10)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
#         output2 = self.last_layer2(output)
        return output

    def predict(self, test_x):
        with torch.no_grad():

            # The output of the model is a multitask MVN, where both the data points
            # and the tasks are jointly distributed
            # To compute the marginal predictive NLL of each data point,
            # we will call `to_data_independent_dist`,
            # which removes the data cross-covariance terms from the distribution.
            preds = model.likelihood(model(test_x)).to_data_independent_dist()

        return preds.mean.mean(0), preds.variance.mean(0)




model = MultitaskDeepGP(x1.shape)
model = model
likelihood=model.likelihood
likelihood = likelihood

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
mll = DeepApproximateMLL(VariationalELBO(likelihood, model, num_data=y1.size(0)))

num_epochs = 1 if smoke_test else 500
epochs_iter = num_epochs
for i in range(epochs_iter):
    optimizer.zero_grad()
    output = model(x1.float())
    loss = -mll(output, y1)
    
    loss.backward()
    optimizer.step()

model.eval()
likelihood.eval()

train_y1=y1[:,0]
train_y2=y1[:,1]
yp=np.concatenate((train_y1.reshape(train_y1.shape[0],1),train_y2.reshape(train_y2.shape[0],1)),axis=1)
yp_query=yp.reshape(train_y1.shape[0],2)

N_obj=2
#data_x=np.concatenate((data_x,x_query.reshape(1,N_dim)),axis=0)
data_yp=yp_query



#y_pareto_truth,ind=eng.Pareto_finder_py(matlab.double(data_y),nargout=2)
#y_pareto_truth=np.asarray(y_pareto_truth)
y_pareto_truth,ind = Pareto_finder(data_yp,goal)
#ind=np.asarray(ind)
#ind=ind.astype(int)
#ind=1
#x_pareto_truth=data_x[ind]
#hv_t=eng.HV(matlab.double(y_pareto_truth),nargout=1)
#hv_t=np.asarray(hv_t).reshape(1,1)
#     hv_t = (HV_Calc(goal,ref,y_pareto_truth)).reshape(1,1)
hv_truth=(HV_Calc(goal,ref,y_pareto_truth)).reshape(1,1)

test_x=torch.from_numpy((dirlet5D(500,5))/32)
# test_x = lhs(N_dim,100)
# test_x=torch.from_numpy(normalize(test_x, axis=1, norm='l1'))
# test_x2 = test_x
# test_x3 = test_x
# test_x4 = test_x
for k in range(500):

    print("****************",k,"********************")
    #test_x2 = torch.linspace(0, 1, 51)
    #torch.cuda.empty_cache()
    torch.set_flush_denormal(True)
    xi=0.9
    #test_x = torch.linspace(0, 1, 51)







#         test_x = lhs(N_dim,100)
#         test_x=torch.from_numpy(normalize(test_x, axis=1, norm='l1'))
#         test_x=torch.from_numpy((dirlet5D(100,5))/32)
#         with torch.no_grad(), gpytorch.settings.fast_pred_var():
# #             test_x = lhs(N_dim,100)
# #             test_x=torch.from_numpy(normalize(test_x, axis=1, norm='l1')).cuda()
#             test_x=torch.from_numpy((dirlet5D(100,5))/32).cuda()
#             mean_t, var_t = model.predict(test_x.float())
#             lower = mean_t - 2 * var_t.sqrt()
#             upper = mean_t + 2 * var_t.sqrt()


    e=True
    counter2=0
#         while(e and (counter2<10)):

#     test_x = lhs(N_dim,500)
#             try:
#                 if (counter2>3):
#                     test_x=torch.from_numpy((dirlet5D(100,5))/32)
#                 else:

    test_x=torch.from_numpy((dirlet5D(500,5))/32)
    for i in range(int(test_x.shape[0]/5)):


        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            ind=i*5
            if i ==0:
                test_xt=test_x[0:5,:]
                mean, var = model.predict(test_xt.float())
                mean_t=mean
                var_t=var
            else:
                test_xt=test_x[ind:ind+5,:]
                mean, var = model.predict(test_xt.float())
                mean_t=torch.cat((mean_t,mean),0)
                var_t=torch.cat((var_t,var),0)

    test_xt=test_x[ind+5:,:]
#         print(test_xt.float())
    if test_xt.numel()!=0:
        mean, var = model.predict(test_xt.float())
        mean_t=torch.cat((mean_t,mean),0)
        var_t=torch.cat((var_t,var),0)


    task=0

#         lower = mean_t - 2 * var_t.sqrt()
#         upper = mean_t + 2 * var_t.sqrt()




    mean_t1=mean_t[:, 0]
    #print(mean_t1.shape,"%%%%%%%%%%%%%%%%%%%%%Debug%%%%%%%%%%%%%%%%%%%%%")
    mean_t2=mean_t[:, 1]
    mean_t3=mean_t[:, 2]
    #print(mean_t2.shape,"%%%%%%%%%%%%%%%%%%%%%Debug%%%%%%%%%%%%%%%%%%%%%")
    var_t1=var_t[:, 0].sqrt()
    var_t2=var_t[:, 1].sqrt()
    var_t3=var_t[:, 1].sqrt()
    eh_mean=np.concatenate((np.array([mean_t1.numpy()]).T,np.array([mean_t2.numpy()]).T),axis=1)
    eh_std=np.concatenate((np.array([var_t1.numpy()]).T,np.array([var_t2.numpy()]).T),axis=1)
    
    dgp_mean1_lst.append(mean_t1.numpy())
    dgp_mean2_lst.append(mean_t2.numpy())
    dgp_mean3_lst.append(mean_t3.numpy())
    dgp_std1_lst.append(var_t1.numpy())
    dgp_std2_lst.append(var_t2.numpy())
    dgp_std3_lst.append(var_t3.numpy())
    
    
    n_jobs=multiprocessing.cpu_count()
    def calc(ii):
        e = EHVI(eh_mean[ii],eh_std[ii],goal,ref,y_pareto_truth)
        return e

    ehvi=Parallel(n_jobs)(delayed(calc)([jj]) for jj in range(eh_mean.shape[0]))
    ehvi=np.array(ehvi)

    x_star=np.argmax(ehvi)

#     kld_mv = np.log(std2/std1) + (std1**2 + (mean1 - mean2)**2)/(2*std2**2) - 0.5
#                 max_val, x_star, EI = expected_improvement(np.max(y1[:,0].detach().cpu().numpy()), xi,mean_t[:, task].cpu().numpy(),var_t[:, task].sqrt().cpu().numpy() )#(upper[:, task].cpu().numpy()-mean_t[:, task].cpu().numpy())/2

    new_x=test_x.detach()[x_star]
    #new_x=torch.rand(100)[x_star]

#         if b==2:
#             dgp_pred_lst.append(mean_t[:, task].cpu().numpy())
#             dgp_std_lst.append(var_t[:, task].sqrt().cpu().numpy() )
#             dgp_EI_lst.append(EI)
#             #gp_query_lst()
#             dgp_test_lst.append(test_x)
#             dgp_query_lst.append(new_x)
#         test_x = torch.cat((test_x[:x_star],test_x[x_star+1:]))
#     train_y1 = train_x1 ** (2)
#     train_y2 = (train_x2 **(2))*(-1)
    new_y=bmft(new_x.unsqueeze(0))
    new_y2=cft(new_x.unsqueeze(0))
    new_y3=c2ft(new_x.unsqueeze(0))


#     new_y= torch.cos(new_x * (2 * math.pi))# + torch.randn(new_x.size()) * 0.2

    data_x=np.concatenate((x1.detach().numpy(), np.array([new_x.numpy()])),axis=0)
    data_y=np.concatenate((y1[:, 0].detach().numpy(),new_y.numpy()),axis=0)
    x1_t=torch.tensor(data_x)
    train_y2_t=torch.tensor(data_y)
#     data=np.concatenate((train_x1.detach().numpy(),new_x.numpy().reshape(1)),axis=0)
    data_y=np.concatenate((y1[:, 1].detach().numpy(),new_y2.numpy()),axis=0)
#     train_x3=torch.tensor(data_x)
    train_y3_t=torch.tensor(data_y)
#     data_x=np.concatenate((train_x1[:, 2].detach().numpy(),new_x.numpy().reshape(1)),axis=0)
    data_y=np.concatenate((y1[:, 2].detach().numpy(),new_y3.numpy()),axis=0)
#         train_x4=torch.tensor(data_x)
    train_y4_t=torch.tensor(data_y)
#     data_x=np.concatenate((train_x1[:, 3].detach().numpy(),new_x.numpy().reshape(1)),axis=0)
#     data_y=np.concatenate((train_y1[:, 3].detach().numpy(),new_y4.numpy().reshape(1)),axis=0)
# #     train_x5=torch.tensor(data_x)
#     train_y5=torch.tensor(data_y)


    y1_t = torch.stack([
        train_y2_t,
        train_y3_t,
        train_y4_t,

    ], -1)

    train_y1=y1_t[:,0]

    train_y2=y1_t[:,1]
    train_y3=y1_t[:,2]
    yp=np.concatenate((train_y1.reshape(train_y1.shape[0],1),train_y2.reshape(train_y2.shape[0],1)),axis=1)
    yp_query=yp.reshape(yp.shape[0],2)
    #print(yp_query)
    N_obj=2
    #data_x=np.concatenate((data_x,x_query.reshape(1,N_dim)),axis=0)
    data_yp=yp



    #y_pareto_truth,ind=eng.Pareto_finder_py(matlab.double(data_y),nargout=2)
    #y_pareto_truth=np.asarray(y_pareto_truth)
    y_pareto_truth,ind = Pareto_finder(data_yp,goal)
    #ind=np.asarray(ind)
    #ind=ind.astype(int)
    #ind=1
    #x_pareto_truth=data_x[ind]
    #hv_t=eng.HV(matlab.double(y_pareto_truth),nargout=1)
    #hv_t=np.asarray(hv_t).reshape(1,1)
    hv_t = (HV_Calc(goal,ref,y_pareto_truth)).reshape(1,1)
    hv_truth=np.concatenate((hv_truth,hv_t.reshape(1,1)))

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




    #model = MultitaskGPModel((full_train_x, full_train_i), full_train_y, likelihood)
    #model = MultitaskDeepGP(x1.shape)
    model = MultitaskDeepGP(x1_t.shape)
    model = model
    likelihod=model.likelihood
    likelihood = likelihood
    
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.9)
    mll = DeepApproximateMLL(VariationalELBO(likelihood, model, num_data=y1.size(0)))

    num_epochs = 1 if smoke_test else 200
    
    epochs_iter=range(num_epochs)
    for i in epochs_iter:
        optimizer.zero_grad()
        output = model(x1_t.float())
        loss = -mll(output, y1_t)
        
        loss.backward()
#             print('Iter {}/{} - Loss: {}   lengthscale: {} lengthscale: {}  noise: {}'.format(
#                 i + 1, num_epochs, loss.item(),
#                 model.hidden_layer.covar_module.base_kernel.lengthscale,
#                 model.last_layer.covar_module.base_kernel.lengthscale,

#                 model.likelihood.noise
#             ))
        optimizer.step()
    #print("yooooooooooooooooooooooooo")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = DeepApproximateMLL(VariationalELBO(likelihood, model, num_data=y1.size(0)))

    num_epochs = 1 if smoke_test else 200
    
    epochs_iter=range(num_epochs)
    for i in epochs_iter:
        optimizer.zero_grad()
        output = model(x1_t.float())
        loss = -mll(output, y1_t)
        
        loss.backward()
#             print('Iter {}/{} - Loss: {}   lengthscale: {} lengthscale: {}  noise: {}'.format(
#                 i + 1, num_epochs, loss.item(),
#                 model.hidden_layer.covar_module.base_kernel.lengthscale,
#                 model.last_layer.covar_module.base_kernel.lengthscale,

#                 model.likelihood.noise
#             ))
        optimizer.step()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = DeepApproximateMLL(VariationalELBO(likelihood, model, num_data=y1.size(0)))

    num_epochs = 1 if smoke_test else 100
    
    epochs_iter=range(num_epochs)
    for i in epochs_iter:
        optimizer.zero_grad()
        output = model(x1_t.float())
        loss = -mll(output, y1_t)
        
        loss.backward()
#             print('Iter {}/{} - Loss: {}   lengthscale: {} lengthscale: {}  noise: {}'.format(
#                 i + 1, num_epochs, loss.item(),
#                 model.hidden_layer.covar_module.base_kernel.lengthscale,
#                 model.last_layer.covar_module.base_kernel.lengthscale,

#                 model.likelihood.noise
#             ))
        optimizer.step()
        e=False

#             except Exception as error:
#                 print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  EXCEPTION %%%%%%%%%%%%%%%%%%%",error)
#                 e=True

#             counter2=counter2+1

    x1=deepcopy(x1_t)
    y1=deepcopy(y1_t)
    train_y2=deepcopy(train_y2_t)
#     data_x=np.concatenate((train_x1.detach().numpy(),new_x.numpy().reshape(1)),axis=0)

    train_y3=deepcopy(train_y3_t)

    train_y4=deepcopy(train_y4_t)
#         hv_truth=hv_truth_t
    model.eval()
    likelihood.eval()
    #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #         test_x = lhs(N_dim,100)
    #         test_x=torch.from_numpy(normalize(test_x, axis=1, norm='l1'))
    #         mean, var = model.predict(test_x.float())
    #         lower = mean - 2 * var.sqrt()
    #         upper = mean + 2 * var.sqrt()
#         if b==2:
#             dgp_y_lst.append(y1[:,0].detach().cpu().numpy())
#             dgp_x_lst.append(x1.detach().cpu().numpy())
#         if counter2!=100:    
#             dgp_max_tmp[0]=y1[:,0].detach().cpu().numpy()
    
    
    #break

import pickle
with open(file_out+'-hv.pl', 'wb') as handle:
        pickle.dump(hv_truth, handle, protocol=pickle.HIGHEST_PROTOCOL) 
with open(file_out+'-queryx1.pl', 'wb') as handle:
        pickle.dump(x1, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out+'-queryx2.pl', 'wb') as handle:
        pickle.dump(x1, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
with open(file_out+'-queryx3.pl', 'wb') as handle:
        pickle.dump(x1, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
with open(file_out+'-queryy1.pl', 'wb') as handle:
        pickle.dump(train_y2, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out+'-queryy2.pl', 'wb') as handle:
        pickle.dump(train_y3, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out+'-queryy3.pl', 'wb') as handle:
        pickle.dump(train_y4, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
with open(file_out+'-mean1.pl', 'wb') as handle:
        pickle.dump(dgp_mean1_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out+'-mean2.pl', 'wb') as handle:
        pickle.dump(dgp_mean2_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
with open(file_out+'-mean3.pl', 'wb') as handle:
        pickle.dump(dgp_mean3_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
with open(file_out+'-std1.pl', 'wb') as handle:
        pickle.dump(dgp_std1_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out+'-std2.pl', 'wb') as handle:
        pickle.dump(dgp_std2_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_out+'-std3.pl', 'wb') as handle:
        pickle.dump(dgp_std3_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)