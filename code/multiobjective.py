# -*- coding: utf-8 -*-
"""

@author: Danial Khatamsaz
"""

 # means : GP mean estimation of objectives of the test points (fused means in
 # multifidelity cases). Each column for 1 objective values

 # sigmas : uncertainty of GP mean estimations (std). Each column for 1 objective

 # goal : a row vector to define which objectives to be minimized or
 # maximized. zero for minimizing and 1 for maximizing. Example: [ 0 0 1 0 ... ]

 # ref : hypervolume reference for calculations

 # pareto : Current true pareto front obtained so far

 # Note that in all variables, the order of columns should be the
 # same. For example, the 1st column of all matrices above is
 # related to the objective 1. Basically, each row = 1 design
 
 ## All the inputs should be 2D np arrays

import numpy as np
import pandas as pd
from scipy.stats import norm
#from graph import improvement , volume
from copy import deepcopy

def EHVI(mm,ss,goal,ref,pareto):
    means=deepcopy(mm)
    sigmas=deepcopy(ss)
    N_obj = means.shape[1]
    for i in range(goal.shape[1]):
        if goal[0,i]==1:
            means[:,i]=-1*means[:,i]
            pareto[:,i]=-1*pareto[:,i]
    
#  Sorting the non_dominated points considering the first objective
#  It does not matter which objective to sort but lets do it with the
#  1st objective
    pareto=pareto[pareto[:, 0].argsort()]
    ehvi=np.array([])
    for i in range(means.shape[0]):
        hvi=np.array([0])
        box=np.array([1])
        # EHVI over the box from infinity to the ref point
        for j in range(N_obj):
            s = (ref[0,j]-means[i,j])/sigmas[i,j]
            box = box*((ref[0,j]-means[i,j])*norm.cdf(s)+sigmas[i,j]*norm.pdf(s))
            
        # calculate how much adding a test point can improve the hypervolume
        hvi = recursive(means[i,:].reshape(1,N_obj),sigmas[i,:].reshape(1,N_obj),ref,pareto)
        ehvi = np.append (ehvi, [box-hvi])
        
    return ehvi
        
def recursive(means,sigmas,ref,pareto):
    N_obj=pareto.shape[1]
    if pareto.shape[0]==1:
        hvi_temp=np.array([1])
        for j in range(N_obj):
            s_up=(ref[0,j]-means[0,j])/sigmas[0,j]
            s_low=(pareto[0,j]-means[0,j])/sigmas[0,j]
            up = ((ref[0,j]-means[0,j])*norm.cdf(s_up)+sigmas[0,j]*norm.pdf(s_up))
            low = ((pareto[0,j]-means[0,j])*norm.cdf(s_low)+sigmas[0,j]*norm.pdf(s_low))
            hvi_temp = hvi_temp*(up-low)
        improvement = hvi_temp

    else:

        hvi_temp=np.array([1])
        for j in range(N_obj):
            s_up=(ref[0,j]-means[0,j])/sigmas[0,j]
            s_low=(pareto[0,j]-means[0,j])/sigmas[0,j]
            up = ((ref[0,j]-means[0,j])*norm.cdf(s_up)+sigmas[0,j]*norm.pdf(s_up))
            low = ((pareto[0,j]-means[0,j])*norm.cdf(s_low)+sigmas[0,j]*norm.pdf(s_low))
            hvi_temp = hvi_temp*(up-low)
            
        pareto_prime=np.zeros([1,N_obj])
        for z in range(1,pareto.shape[0]):
            temp = np.max([pareto[0,:],pareto[z,:]],axis=0)
            pareto_prime = np.append(pareto_prime,[temp],axis=0)
            
        pareto_prime=np.delete(pareto_prime,0,axis=0)
        pareto_prime,null=Pareto_finder(pareto_prime,np.zeros([1,N_obj]))
        hvi_temp2 = recursive(means,sigmas,ref,pareto_prime)
        pareto=np.delete(pareto,0,axis=0)
        improve2 = recursive(means,sigmas,ref,pareto)
        improvement = improve2 + hvi_temp - hvi_temp2
    
    return improvement
        

def Pareto_finder(VV,goal):
# V is a matrix, each row is the objectives of one design points
# goal : a row vector to define which objectives to be minimized or
# maximized. zero for minimizing and 1 for maximizing. Example: [ 0 0 1 0 ... ]
    V=deepcopy(VV)
    for i in range(goal.shape[1]):
        if goal[0,i]==1:
            V[:,i]=V[:,i]*-1
            
    pareto=np.zeros([1,goal.shape[1]])
    ind=np.zeros([1])
    
    for i in range(V.shape[0]):
        p=V[i,:]
        s=V
        s=np.delete(s,i,axis=0)
        trig=0
        for j in range(s.shape[0]):
            temp=p-s[j,:]
            if np.min(temp)>=0:
                trig=1 #means vector p is dominated
        if trig==0:
            pareto=np.append(pareto,[p],axis=0)
            ind=np.append(ind,i)
    
    
    pareto=np.delete(pareto,0,axis=0)
    ind=np.delete(ind,0)
    
    # Changing back the signs if were changed before
    for i in range(goal.shape[1]):
        if goal[0,i]==1:
            pareto[:,i]=-1*pareto[:,i]
    ind = [ int(x) for x in ind ]
    return pareto,ind


def HV_Calc(goal,ref,pp):
# this function calculates the hypervolume

# goal : a row vector to define which objectives to be minimized or
# maximized. zero for minimizing and 1 for maximizing. Example: [ 0 0 1 0 ... ]

# ref : hypervolume reference for calculations

# pareto : Current true pareto front obtained so far

# Note that in all variables, the order of columns should be the
# same. For example, the 1st column of all matrices above is
# related to the objective 1. Basically, each row = 1 design
# 
    pareto=deepcopy(pp)
    N_obj=pareto.shape[1]
    
    for i in range(goal.shape[1]):
        if goal[0,i]==1:
            pareto[:,i]=-1*pareto[:,i]
    
    pareto[pareto[:, 0].argsort()]
    
    hv=recursive_HV(ref,pareto)
    
    return hv

def recursive_HV(ref,pareto):
    N_obj=pareto.shape[1]
    if pareto.shape[0]==1:
        hv_temp=np.array([1])
        for j in range(N_obj):
            length = ref[0,j]-pareto[0,j]
            hv_temp=hv_temp*length
        hypervolume=hv_temp 
    else:
        hypervolume=np.array([0])
        hv_temp=np.array([1])
        pareto_prime=np.zeros([1,N_obj])
        
        for j in range(N_obj):
            length=ref[0,j]-pareto[0,j]
            hv_temp=hv_temp*length
        for z in range(1,pareto.shape[0]):
            temp = np.max([pareto[0,:],pareto[z,:]],axis=0)
            pareto_prime = np.append(pareto_prime,[temp],axis=0)
            
        pareto_prime=np.delete(pareto_prime,0,axis=0)
        
        pareto_prime,null=Pareto_finder(pareto_prime,np.zeros([1,N_obj]))
        hv_temp2 = recursive_HV(ref,pareto_prime)
        pareto=np.delete(pareto,0,axis=0)
        improve2 = recursive_HV(ref,pareto)
        hypervolume = improve2 + hv_temp - hv_temp2
    
    return hypervolume
    
    
          
        

    
    
    






