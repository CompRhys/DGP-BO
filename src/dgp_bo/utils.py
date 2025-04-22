import numpy as np
import pandas as pd
import torch


def lookup_in_h5_file(x, filename, column_name="tec"):
    xt = x.cpu().numpy()
    c11 = []
    conc = np.asarray([np.asarray([i, j, k, l, m]) for i, j, k, l, m in xt])
    for p, q, r, s, t in conc:
        df1 = pd.read_hdf(filename)
        c = df1.loc[
            (df1["Fe"] == p)
            & (df1["Cr"] == q)
            & (df1["Ni"] == r)
            & (df1["Co"] == s)
            & (df1["Cu"] == t)
        ][column_name].to_numpy()[0]
        c11.append(c)

    return torch.from_numpy(np.asarray(c11))


def bmft(x, filename, bmf="tec"):
    xt = x.cpu().numpy()
    c11 = []
    conc = np.asarray([np.asarray([i, j, k, l, m]) for i, j, k, l, m in xt])
    for p, q, r, s, t in conc:
        df1 = pd.read_hdf(filename)
        c = df1.loc[
            (df1["Fe"] == p)
            & (df1["Cr"] == q)
            & (df1["Ni"] == r)
            & (df1["Co"] == s)
            & (df1["Cu"] == t)
        ][bmf].to_numpy()[0]
        c11.append(c)

    return torch.from_numpy(np.asarray(c11))


def cft(x, filename, bmf="bulkmodul_eq"):
    xt = x.cpu().numpy()
    c11 = []
    conc = np.asarray([np.asarray([i, j, k, l, m]) for i, j, k, l, m in xt])
    for p, q, r, s, t in conc:
        df1 = pd.read_hdf(filename)
        c = df1.loc[
            (df1["Fe"] == p)
            & (df1["Cr"] == q)
            & (df1["Ni"] == r)
            & (df1["Co"] == s)
            & (df1["Cu"] == t)
        ][bmf].to_numpy()[0]
        c11.append(c)

    return torch.from_numpy(np.asarray(c11))


def c2ft(x, filename, bmf="volume_eq"):
    xt = x.cpu().numpy()
    c11 = []
    conc = np.asarray([np.asarray([i, j, k, l, m]) for i, j, k, l, m in xt])
    for p, q, r, s, t in conc:
        df1 = pd.read_hdf(filename)
        c = df1.loc[
            (df1["Fe"] == p)
            & (df1["Cr"] == q)
            & (df1["Ni"] == r)
            & (df1["Co"] == s)
            & (df1["Cu"] == t)
        ][bmf].to_numpy()[0]
        c11.append(c)

    return torch.from_numpy(np.asarray(c11))
