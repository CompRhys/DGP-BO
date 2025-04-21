"""Created on Wed Jun 14 22:17:58 2023.

@author: Dania
"""

import numpy as np
import pandas as pd
import scipy
import scipy.linalg
from scipy.linalg import block_diag


def Kernel(dataset1, dataset2, sf, l):
    S1 = dataset1.shape[0]
    S2 = dataset2.shape[0]
    K = np.zeros([S1, S2])

    for i in range(S2):
        temp = ((dataset1 - dataset2[i]) ** 2) / l**2
        temp_sum = -0.5 * np.sum(temp, axis=1)
        K[:, i] = sf * np.exp(temp_sum)

    return K


def MeanCov1(xtrain, ytrain, l, sf, sn, xtest):
    N_task = ytrain.shape[1]

    df = pd.DataFrame(ytrain)
    dff = df.corr(method="pearson")
    Kf = np.array(dff)
    # print(Kf)
    # Kf=np.eye(N_task)

    Kxz = np.kron(Kf, Kernel(xtrain, xtest, sf, l))
    Kzz = np.kron(Kf, Kernel(xtest, xtest, sf, l))
    # Kzx=np.transpose(Kxz)

    B = block_diag(np.diag(sn[:, 0]))
    for i in range(1, N_task):
        B = block_diag(B, np.diag(sn[:, i]))

    B = B**2

    # B=block_diag(np.diag(sn[:,0]),np.diag(sn[:,1]))**2

    GPKxx = np.kron(Kf, Kernel(xtrain, xtrain, sf, l))

    H = GPKxx + B

    R = scipy.linalg.cholesky(H, lower=False)

    ytrain_flat = ytrain.flatten("F")
    ytrain_flat = ytrain_flat.reshape(len(ytrain_flat), 1)

    ad = np.linalg.solve(np.transpose(R), ytrain_flat)
    bd = np.linalg.solve(np.transpose(R), Kxz)

    m = np.transpose(bd) @ ad
    co = Kzz - np.transpose(bd) @ bd

    means = m.reshape(N_task, -1).T
    vari = (np.diag(co)).reshape(N_task, -1).T

    return means, vari
