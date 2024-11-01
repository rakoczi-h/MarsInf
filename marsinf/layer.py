import numpy as np
from datetime import datetime

from marsinf.utils import multivariate, matern_covariance, great_circle_distance

class Layer():
    """
    Class that can store information about planetary layers and containg methods to simulate density distributions of the layer.
    Parameters:
    ----------
        parameters: dict
            The parameters of the planet layer. Usually contains information about the aprameters of the density distribution. Each element of the dictionary is also separately saved as an attribute to the class. If provided at initialisation it overwrites every other default or input value for these attributes. (Default: None)
            If None, default values of these parameters are used:
                "epsilon": None
                "var : None [kg/m^3], the variance of the density distribution
                "kappa": None
                "av_dens": None [kg/m^3], the avarage density
        matern: np.ndarray
            Has shape N x N, where N=no. of points in latitude x no.of points in longitude. The covariance function that defines the multivariate distribution of the densities in the layer. (Default: None)
        dens_model: np.ndarray
            1D array containing density values, describing the lateral density variation in this layer. Has shape N. Units: kg/m^3. (Default: None).
        topo_model: np.ndarray
            1D array containing the topography elevation values. Has shape N. Units: m. (Default: None).
        lat, long: np.ndarrays
            These are assumed to be created with meshgrid, hence these are 2D arrays with shape [no. of points in latitude grid, no. of points in longitude grid] and reversed, respectively. The shape parameter is inferred using lat. Units: degrees (Default: None)
        psi: np.ndarray
            The great circle distances between every single pair of points in the lat-long grid. The shape is [N x N], where N=no. of points in latitude x no.of points in longitude. (Default: None)
        resolution: list
            The resolution in [latitude, longitude] in degrees. (Default: None)
    """
    def __init__(self, parameters=None, matern=None, dens_model=None, topo_model=None, lat=None, long=None, psi=None, resolution=None):
        self.parameter_labels = None
        self.parameters = parameters
        self.lat = lat # assumed that it's in degrees
        self.long = long
        self.coordinates = None
        if self.lat is not None and self.long is not None:
            self.coordinates = np.c_[self.lat.flatten(), self.long.flatten()]
        self.matern = matern
        self.dens_model = dens_model
        self.topo_model = topo_model
        self.resolution = resolution
        self.psi = psi


    def __setattr__(self, name, value):
        if name == 'parameters':
            if value is not None:
                if not isinstance(value, dict):
                    raise ValueError('parameters has to be a dictionary')
                if self.parameter_labels is None:
                    self.parameter_labels = list(value.keys())
                for i, key in enumerate(self.parameter_labels):
                    super().__setattr__(key, value[key])
                value.setdefault("epsilon", None)
                value.setdefault("var", None)
                value.setdefault("kappa", None)
        super().__setattr__(name,value)

    def matern_covariance(self, verbose=False):
        """
        Makes a matern covariance matrix with the parameters taken from the class.
        Parameters
        ----------
            verbose: bool
                If True, the time taken to generate the covariance function is printed in the end.
        Output
        ------
            matern: np.ndarray
                The matern covariance function, which is the same size as psi.
        """
        if self.psi is None:
            if self.coordinates is None:
                raise ValueError('Need to give either psi or coordinates attributes to the class')
            else:
                self.psi = great_circle_distance(lat=self.coordinates[:,0]/180.0*np.pi, long=self.coordinates[:,1]/180.0*np.pi)
        if None in [self.epsilon, self.kappa, self.var]:
            raise ValueError('Set the epsilon, kappa, and var attributes of the class')
        self.matern = matern_covariance(self.psi, self.epsilon, self.kappa, self.var, timed=verbose)
        return self.matern

    def make_dens_model(self, seed=None, verbose=False):
        """
        Creates a spatial density model for the layer by sampling a multivariate gaussian distribution.
        Parameters
        ----------
            seed: int
                If given, this seed number is passed to the multivariate sampling function, for repeatability. (Default: None)
            verbose: bool
                If True, the time taken to generate the distribution is printed in the end.
        Output
        ------
            dens_model: np.ndarray
                1D array containing density values, describing the lateral density variation in this layer. Has shape N. N=no. of points in latitude x no.of points in longitude. (Default: None).
        """
        if self.matern is None: # making matern covariance function if doesn't exist yet
            self.matern_covariance()
        self.dens_model = multivariate(self.matern, self.av_dens*np.ones(np.shape(self.matern)[0]), seed=seed, timed=verbose)
        return self.dens_model

    # Plotting tools
    def plot_layer(self, filename='layer.png'):
        """
        Creates a plot of the density distribution of the layer.
        Parameters
        ----------
            filename: str
                Specifies where the figure is saved.
        """
        fig, ax = plt.subplots(1,1,subplot_kw={'projection': 'mollweide'})
        im = ax.pcolormesh(self.lat, self.long, np.reshape(self.dens_model, self.resolution), cmap=plt.cm.jet)
        plt.savefig(filename)
        plt.close()
