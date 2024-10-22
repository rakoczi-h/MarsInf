import numpy as np
import scipy.special
from datetime import datetime
import scipy.linalg
import matplotlib.pyplot as plt

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def make_long_lat(resolution, ranges):
    long = np.arange(ranges[0][0], ranges[0][1], resolution[1])
    lat = np.arange(ranges[1][0], ranges[1][1], resolution[0])
    Long, Lat = np.meshgrid(long, lat)
    return Long, Lat


def great_circle_distance(long, lat, timed=False):
    """
    Calculates the great circle distance between each pair of points.
    Output has the shape of nxm where n is the lenght of long and m is the length of lat.
    """
    start = datetime.now()
    delta_long = np.subtract.outer(long, long)
    prod_1 = np.multiply.outer(np.sin(lat), np.sin(lat))
    prod_2 = np.multiply.outer(np.cos(lat), np.cos(lat))
    dist = prod_1+(prod_2*np.cos(delta_long))
    y = np.ones(np.shape(dist))
    dist = np.where(dist>1, y, dist)
    y = -1*np.ones(np.shape(dist))
    dist = np.where(dist<-1, y, dist)
    central_angle = np.arccos(dist)
    if timed:
        print(f"Time taken by great circle distance calculation: {datetime.now()-start}")
    return central_angle

def multivariate(cov, mean, size=None, seed=None, cholesky=True, timed=False):
    """
    Parameters
    ----------
        mean: int, float or np.ndarray
            If it's an array, it has to have the lenght N if the sive of cov in NxN
    """
    np.random.seed(seed=seed)
    start = datetime.now()
    if cholesky:
        x = np.random.normal(loc=0.0, scale=1.0, size=np.shape(cov)[0])
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

def matern_covariance(psi, epsilon, kappa, var, timed=False):
    """
    Makes a matern covariance matrix with the desired parameters. The output is the same size as psi.
    """
    start = datetime.now()
    idx = np.argwhere(psi==0)
    psi[idx[:,0], idx[:,1]] = 1e-32
    var1 = 4*np.sqrt(kappa)/epsilon*np.sin(psi/2)
    bessel = scipy.special.kv(kappa, var1)
    M = (var**2)*(2**(1-kappa))/scipy.special.gamma(kappa)*((var1)**kappa)*bessel
    matern = M + 1e-5*np.identity(np.shape(M)[0]) # adding small value to the diagonal to ensure the matrix is positive definite
    if timed:
        print(f"Time taken by matern_covariance: {datetime.now()-start}")
    return matern


