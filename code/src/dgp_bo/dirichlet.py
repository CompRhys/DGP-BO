import numpy as np
from scipy.stats import dirichlet


def pos(x):
    if x >= 0:
        return x
    if x < 0:
        return 0
    return None


def dirlet5D(N_samp, dim):
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
                        32 - round(i * 32) - round(j * 32) - round(k * 32) - round(l * 32)
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
            # print("********************exception**********")

    return np.asarray(out)
