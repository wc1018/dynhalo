import numpy as np

def biaspb_model(
    nu_orb_edges: np.ndarray,
    delta_sc0: float,
    b: float,
    *args,
) -> float:
    """Compute the bias prediction from peak-background split.

    Args:
        nu_orb_edges: 1D array with edges of bins in peak height (deltasc=1.686)
        theta: 1D list or array with model parameters
        args: tuple containing utilities or other information
    """
    delta_sc = 1.686
    nu_orb, ids = args

    nu0 = nu_orb * delta_sc0 / delta_sc
    nuprime = nu0 * (1 + b * np.exp(-nu0) / delta_sc0)

    bias = 1. + (nuprime**2 - 1) / (delta_sc0 + b * np.exp(-nu0))
    mean_bias = np.zeros(nu_orb_edges.size - 1)
    for i in range(mean_bias.size):
        mean_bias[i] = np.mean(bias[ids[i]])

    return mean_bias


if __name__ == '__main__':
    pass
