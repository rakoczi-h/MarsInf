import numpy as np
import scipy.special
from datetime import datetime
import scipy.linalg

def is_pos_def(x):
    """
    Checks whether the matrix x is positive-definite
    Parameters:
    ----------
        x : np.ndarray
    Output:
    ------
        bool
    """
    return np.all(np.linalg.eigvals(x) > 0)

def get_colors(n):
    return ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]

def make_long_lat(resolution, ranges):
    """
    Makes a longitude and langitude meshgrid based on required ranges and resolution.
    Parameters:
    ----------
        resolution: list
            [[minimum longitude, maximum longitude], [minimum latitude, maximum latitude]]
        ranges: list
            [resolution in latitude, resolution in longitude]
    Output:
    ------
        Long, Lat: np.ndarrays
            Created by meshgrid.
    """
    long = np.arange(ranges[0][0], ranges[0][1], resolution[1])
    lat = np.arange(ranges[1][0], ranges[1][1], resolution[0])
    Long, Lat = np.meshgrid(long, lat)
    return Long, Lat


def great_circle_distance(long, lat, timed=False):
    """
    Calculates the great circle distance between each pair of points.
    Parameters
    ---------
        long: np.ndarray
            Has to have the shape n x m where n is the number of grid points in longitude, and n is the number of grid points in latitude. Units: radians.
        lat: np.ndarray
            Has to have the shape n x m where n is the number of grid points in longitude, and n is the number of grid points in latitude. Units: radians.
        timed: bool
            If True, the time taken by the function is printed in the end. (Default: False)
    Output:
    ------
        central_angle: np.ndarray
            Output has the shape of [n x m, n x m] where n is the lenght of long and m is the length of lat.
    """
    start = datetime.now()
    delta_long = np.subtract.outer(long, long)
    prod_1 = np.multiply.outer(np.sin(lat), np.sin(lat))
    prod_2 = np.multiply.outer(np.cos(lat), np.cos(lat))
    dist = prod_1+(prod_2*np.cos(delta_long))
    # correcting numerical overflow
    y = np.ones(np.shape(dist))
    dist = np.where(dist>1, y, dist)
    y = -1*np.ones(np.shape(dist))
    dist = np.where(dist<-1, y, dist)
    central_angle = np.arccos(dist)
    if timed:
        print(f"Time taken by great circle distance calculation: {datetime.now()-start}")
    return central_angle

def multivariate(cov, mean, size=1, seed=None, cholesky=True, timed=False):
    """
    Parameters
    ----------
        mean: int, float or np.ndarray
            If it's an array, it has to have the lenght N if the size of cov in NxN
        size: int
            The number of realisations we want to generate from the distribution. Each sample has size N. (Default: 1)
        seed: int
            If None, no seed is used. (Default: None)
        cholesky: bool
            If True, the cholesky decomposition trick is used to generate samples from a multivariate. Otherwise the np.random.multivariate_normal function is used. (Default: True)
        timed: bool
            If True, the time taken by the sampling is printed in the end.
    Output
    ------
        samples: np.ndarray
            Has shape [size, N]. Samples from the multivaraite distribution defined by the mean and the covariance.
    """
    np.random.seed(seed=seed)
    start = datetime.now()
    if cholesky:
        x = np.random.normal(loc=0.0, scale=1.0, size=(size, np.shape(cov)[0]))
        chol_cov = scipy.linalg.cholesky(cov, lower=False)
        #check = chol_cov @ chol_cov.T.conj()
        samples = np.dot(x, chol_cov)+mean
    else:
        if isinstance(mean, float) or isinstance(mean, int):
            mean = np.ones(np.shape(cov)[0])*mean
        samples = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
    if timed:
        print(f"Time taken by multivariate sampling: {datetime.now()-start}")
    return samples

def matern_covariance(psi, epsilon, kappa, sigma, timed=False):
    """
    Makes a matern covariance matrix with the desired parameters. The chordal matern function is used after Guinness and Fuentes, 2016.
    Parameters
    ----------
        psi: np.ndarray
            The array of great circle distances.
        epsilon: float
            The epsilon parameter, which losely translates to decorrelation distance.
        kappa: float
            The kappa parameter, which controls smoothness
        sigma: float
            The standard deviation.
        timed: bool
            If True, the time taken by the function is printed in the end.
    Output
    ------
        matern: np.ndarray
            The matern covariance matrix with the same shape as psi.
    """
    start = datetime.now()
    idx = np.argwhere(psi==0)
    # Replacing 0 great circle distances with a very small number. This is done so that the bessel function does not return inf values.
    psi[idx[:,0], idx[:,1]] = 1e-32
    var1 = 4*np.sqrt(kappa)/epsilon*np.sin(psi/2)
    bessel = scipy.special.kv(kappa, var1)
    M = (sigma**2)*(2**(1-kappa))/scipy.special.gamma(kappa)*((var1)**kappa)*bessel
    # adding small value to the diagonal to ensure the matrix is positive definite
    matern = M + 1e-5*np.identity(np.shape(M)[0])
    if timed:
        print(f"Time taken by matern_covariance: {datetime.now()-start}")
    return matern


def sections_mean(input_arr, shape):
    """
    Splits the input array into subarrays, and finds the mean of these sections.
    Parameters
    ----------
        input_arr: np.ndarray
            2D array which we aim to down-sample using means of sections.
        shape: list or tuple
            Defines the shape of the output. Has to be consistent with the input.
    Output
    ------
        np.ndarray with shape definted in the input.
    """
    split1 = np.array_split(input_arr, shape[0], axis=0)
    output = np.ones(shape)
    for i, ii in enumerate(split1):
        ii = np.mean(ii, axis=0)
        split2 = np.array_split(ii, shape[1])
        for j, jj in enumerate(split2):
            jj = np.mean(jj)
            output[i,j] = jj
    return output

