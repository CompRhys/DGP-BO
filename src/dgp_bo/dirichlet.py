import numpy as np
from scipy.stats import dirichlet


def pos(x):
    if x >= 0:
        return x
    if x < 0:
        return 0
    return None


def dirichlet5D(n_samples):
    """Draw n_samples from the Dirichlet distribution with 5 dimensions."""
    out = []
    samples = dirichlet.rvs(size=n_samples, alpha=np.ones(5))
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
            out.append(dirichlet5D(n_samples=1)[0])

    return np.asarray(out)
