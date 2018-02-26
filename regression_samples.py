import numpy as np
import numpy.random as random

def simple_sin(inputs):
    """
    A simple sin function with period 1
    """
    return np.sin(2*np.pi*inputs)

def arbitrary_function_1(inputs):
    """
    An arbitrary function to provide an interesting form for regression in 1d
    """
    return inputs*np.sin(2*np.pi*(inputs**2))


def arbitrary_function_2(inputs):
    """
    An arbitrary function to provide an interesting form for regression in 1d
    """
    return np.sin(2*np.pi*(2*inputs-1)**4)


def saw_function(inputs):
    """
    An arbitrary function to provide an interesting form for regression in 1d
    """
    targets = np.empty(inputs.shape)
    targets[inputs<0.5] = inputs[inputs<0.5]
    targets[inputs>=0.5] = inputs[inputs>=0.5]-1
    return targets

def sample_data(N, true_func=None, include_xlim=True, noise=0.3, seed=None):
    """
    Sample 1d input and target data for regression. Produces random inputs
    between 0 and 1 and noise corrupted outputs of the true function.

    Parameters
    ----------
    N - the number of data points to output
    true_func - the true underlying function 
    include_xlim (optional) - whether to include 0 and 1 in the inputs (this can
      improve the stability of your regression results.
    seed (optional) - the seed value (integer) for the pseudo-random number
        generator, allows one to recreate experiments

    Returns
    -------
    inputs - randomly sampled input data (x)
    targets - the associated targets (true_func(x) + gaussian noise)
    """
    if not seed is None:
        np.random.seed(seed)
    # if no underlying function is specified use the first arbitrary function
    # provided above
    if true_func is None:
        true_func = arbitrary_function_1
    # inputs are a collection of N random numbers between 0 and 1
    # for stability we include points at 0 and 1.
    inputs = random.uniform(size=N)
    if include_xlim and N >2:
        inputs[0] = 0
        inputs[-1] = 1
    # outputs are sin(2*pi*x) + gaussian noise
    targets = true_func(inputs) + random.normal(loc=0.0, scale=noise, size=N)
    return inputs, targets


