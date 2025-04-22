import numpy as np
import pandas as pd
import torch


def bmft(x, filename, bmf="tec"):
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
        ][bmf].to_numpy()[0]
        print(c, "****")
        c11.append(c)

    return torch.from_numpy(np.asarray(c11))


def cft(x, filename, bmf="bulkmodul_eq"):
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
        ][bmf].to_numpy()[0]
        c11.append(c)

    return torch.from_numpy(np.array(c11))


def c2ft(x, filename, bmf="volume_eq"):
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
        ][bmf].to_numpy()[0]
        c11.append(c)

    return torch.from_numpy(np.asarray(c11))
