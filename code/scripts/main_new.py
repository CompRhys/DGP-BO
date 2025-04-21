import numpy as np
import pandas as pd
from pyDOE import *
from dgp_bo.multiobjective import Pareto_finder
from sklearn.preprocessing import normalize
from dgp_bo.gpModel import gp_model
import multiprocessing
from joblib import Parallel, delayed

# import matlab.engine
# eng = matlab.engine.start_matlab()
from dgp_bo.MeanCov1 import MeanCov1
from dgp_bo.multiobjective import EHVI, HV_Calc
# objectives: 1-Pugh Ratio B/G , 2-Cauchy Pressure , 3-Yield strength

itr = 301
N_dim = 5
N_test = 1500
N_alt = 300
N_samp = 1
N_obj = 3
ref = np.array([[0, 0, 0]])
goal = np.array([[1, 1, 1]])
opt_imp = []
normal = np.array([[2, 50, 200.0001]])
# normal=torch.tensor(normal)


rep = 100
hv_total = []

for j in range(rep):
    x_init_m1 = pd.DataFrame(pd.read_csv("input_m.csv", header=None)).to_numpy()
    x_init_m2 = pd.DataFrame(pd.read_csv("input_m.csv", header=None)).to_numpy()
    x_init_m3 = pd.DataFrame(pd.read_csv("input_Cur.csv", header=None)).to_numpy()

    y_init_m1 = pd.DataFrame(pd.read_csv("output_Pugh.csv", header=None)).to_numpy()
    y_init_m2 = pd.DataFrame(pd.read_csv("output_Cauchy.csv", header=None)).to_numpy()
    y_init_m3 = pd.DataFrame(pd.read_csv("output_Cur.csv", header=None)).to_numpy()

    y_init_m1 = y_init_m1 / 2.25
    y_init_m2 = y_init_m2 / 63
    y_init_m3 = y_init_m3 / 305

    sf_m1 = 1
    sf_m2 = 1
    sf_m3 = 1

    sn_m = 0.00001

    initial_l = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    GPR_m1 = gp_model(
        x_init_m1,
        y_init_m1.reshape(x_init_m1.shape[0]),
        initial_l,
        sf_m1,
        sn_m,
        N_dim,
        "SE",
        mean=0,
    )
    GPR_m2 = gp_model(
        x_init_m2,
        y_init_m2.reshape(x_init_m2.shape[0]),
        initial_l,
        sf_m2,
        sn_m,
        N_dim,
        "SE",
        mean=0,
    )
    GPR_m3 = gp_model(
        x_init_m3,
        y_init_m3.reshape(x_init_m3.shape[0]),
        initial_l,
        sf_m3,
        sn_m,
        N_dim,
        "SE",
        mean=0,
    )

    # GPR_m1.hp_optimize(update=True)
    # GPR_m2.hp_optimize(update=True)
    # GPR_m3.hp_optimize(update=True)

    # train_x = lhs(N_dim,30)
    # train_x = normalize(train_x, axis=1, norm='l1')
    train_x = pd.DataFrame(pd.read_csv("train_x.csv", header=None)).to_numpy()

    y1, sig1 = GPR_m1.predict_var(train_x)
    y2, sig1 = GPR_m2.predict_var(train_x)
    y3, sig1 = GPR_m3.predict_var(train_x)

    sn_y1 = np.ones([100, 1]) * sn_m  # task1 noise
    sn_y2 = np.ones([100, 1]) * sn_m  # task2 noise
    sn_y3 = np.ones([100, 1]) * sn_m  # task3 noise
    sn = np.concatenate((sn_y1, sn_y2, sn_y3), axis=1)

    y = np.concatenate(
        (y1.reshape(100, 1), y2.reshape(100, 1), y3.reshape(100, 1)), axis=1
    )
    train_y = y.reshape(train_x.shape[0], 3)
    train_y = train_y

    data_x = train_x
    data_y = train_y

    # y_pareto_truth,ind=eng.Pareto_finder_py(matlab.double(data_y),nargout=2)
    # y_pareto_truth=np.asarray(y_pareto_truth)
    y_pareto_truth, index = Pareto_finder(data_y, goal)
    # ind=np.asarray(ind)
    # ind=ind.astype(int)
    # ind=1
    # x_pareto_truth=data_x[ind]

    # hv_t=eng.HV(matlab.double(y_pareto_truth),nargout=1)
    # hv_truth = np.asarray(hv_t).reshape(1,1)
    hv_truth = (HV_Calc(goal, ref, y_pareto_truth)).reshape(1, 1)

    # data_x = pd.DataFrame(pd.read_csv('data_x.csv', header=None)).to_numpy()
    # data_y = pd.DataFrame(pd.read_csv('data_y.csv', header=None)).to_numpy()
    # hv_truth = pd.DataFrame(pd.read_csv('hv_truth.csv', header=None)).to_numpy()
    # y_pareto_truth = pd.DataFrame(pd.read_csv('y_pareto_truth.csv', header=None)).to_numpy()

    iteration = 0

    while iteration < itr:
        iteration = iteration + 1
        x_test = lhs(N_dim, N_test)
        x_test = normalize(x_test, axis=1, norm="l1")
        x_alt = lhs(N_dim, N_alt)
        x_alt = normalize(x_alt, axis=1, norm="l1")

        mean, var = MeanCov1(train_x, train_y, initial_l, 1, sn, x_alt)
        std = var**0.5

        # ehvi = eng.EHVI(matlab.double(mean),matlab.double(std),matlab.double(y_pareto_truth),nargout=1)
        # ehvi = np.asarray(ehvi)
        # ehvi = EHVI(mean,std,goal,ref,y_pareto_truth)

        n_jobs = multiprocessing.cpu_count()

        def calc(ii):
            e = EHVI(mean[ii], std[ii], goal, ref, y_pareto_truth)
            return e

        ehvi = Parallel(n_jobs)(delayed(calc)([jj]) for jj in range(mean.shape[0]))
        ehvi = np.array(ehvi)

        x_star = np.argmax(ehvi)

        # opt_imp.append(max(avg))
        # x_star=np.argmax(avg)
        x_query = x_alt[x_star]
        x_query = x_query.reshape(1, N_dim)
        y1, sig1 = GPR_m1.predict_var(x_query)
        y2, sig1 = GPR_m2.predict_var(x_query)
        y3, sig1 = GPR_m3.predict_var(x_query)
        y = np.concatenate(
            (y1.reshape(1, 1), y2.reshape(1, 1), y3.reshape(1, 1)), axis=1
        )
        y_query = y.reshape(1, 3)
        sn_query = np.array([[sn_m, sn_m, sn_m]])

        data_x = np.concatenate((data_x, x_query.reshape(1, N_dim)), axis=0)
        data_y = np.concatenate((data_y, y_query.reshape(1, N_obj)), axis=0)
        sn = np.concatenate((sn, sn_query), axis=0)
        train_x = data_x
        train_y = data_y

        # y_pareto_truth,ind=eng.Pareto_finder_py(matlab.double(data_y),nargout=2)
        # y_pareto_truth=np.asarray(y_pareto_truth)
        y_pareto_truth, ind = Pareto_finder(data_y, goal)
        # ind=np.asarray(ind)
        # ind=ind.astype(int)
        # ind=1
        # x_pareto_truth=data_x[ind]
        # hv_t=eng.HV(matlab.double(y_pareto_truth),nargout=1)
        # hv_t=np.asarray(hv_t).reshape(1,1)
        hv_t = (HV_Calc(goal, ref, y_pareto_truth)).reshape(1, 1)
        hv_truth = np.concatenate((hv_truth, hv_t.reshape(1, 1)))
        pd.DataFrame(y_pareto_truth).to_csv(
            "/scratch/user/danialkh26/myMTGP_rec/y_pareto_truth.csv",
            header=None,
            index=None,
        )
        # pd.DataFrame(x_pareto_truth).to_csv("/scratch/user/danialkh26/MTGP_Ultimate_seq_myparam/x_pareto_truth.csv", header=None, index=None)
        pd.DataFrame(data_x).to_csv(
            "/scratch/user/danialkh26/myMTGP_rec/data_x.csv", header=None, index=None
        )
        pd.DataFrame(data_y).to_csv(
            "/scratch/user/danialkh26/myMTGP_rec/data_y.csv", header=None, index=None
        )
        pd.DataFrame(hv_truth).to_csv(
            "/scratch/user/danialkh26/myMTGP_rec/hv_truth.csv", header=None, index=None
        )

    hv_total.append(np.ravel(hv_truth))
    pd.DataFrame(hv_total).to_csv(
        "/scratch/user/danialkh26/myMTGP_rec/hv_truth_total.csv",
        header=None,
        index=None,
    )

pd.DataFrame(hv_total).to_csv(
    "/scratch/user/danialkh26/myMTGP_rec/hv_truth_total.csv", header=None, index=None
)
