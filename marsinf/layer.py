import numpy as np
import scipy.special
from datetime import datetime

from marsinf.utils import multivariate, matern_covariance, great_circle_distance

class Layer():
    def __init__(self, parameters=None, depth=None, av_thick=None, av_dens=None, matern=None, dens_model=None, topo_model=None, lat=None, long=None, psi=None, parameter_labels=None, resolution=None):
        self.parameter_labels = parameter_labels
        self.av_thick = av_thick
        self.av_dens = av_dens
        if parameters is None:
            self.parameters = {'var': None, 'kappa': None, 'epsilon': None}
        else:
            self.parameters = parameters
        self.lat = lat # assumed that it's in degrees
        self.long = long
        self.coordinates = None
        if self.lat is not None and self.long is not None:
            self.coordinates = np.c_[self.lat.flatten(), self.long.flatten()]

        self.depth = depth
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
        super().__setattr__(name,value)

    def matern_covariance(self):
        """
        Makes a matern covariance matrix with the desired parameters. The output is the same size as psi.
        """
        if self.psi is None:
            if self.coordinates is None:
                raise ValueError('Need to give either psi or coordinates attributes to the class')
            else:
                print('making psi')
                self.psi = great_circle_distance(lat=self.coordinates[:,0]/180.0*np.pi, long=self.coordinates[:,1]/180.0*np.pi)
        if None in [self.epsilon, self.kappa, self.var]:
            raise ValueError('Set the epsilon, kappa, and var attributes of the class')
        self.matern = matern_covariance(self.psi, self.epsilon, self.kappa, self.var)
        return self.matern

    def make_dens_model(self, seed=None):
        if self.matern is None:
            self.matern_covariance()

        self.dens_model = multivariate(self.matern, self.av_dens*np.ones(np.shape(self.matern)[0]), seed=seed)
        return self.dens_model

    # Plotting tools
    def plot_layer(self, filename='layer.png'):
        fig, ax = plt.subplots(1,1,subplot_kw={'projection': 'mollweide'})
        im = ax.pcolormesh(self.lat, self.long, np.reshape(self.dens_model, self.resolution), cmap=plt.cm.jet)
        plt.savefig(filename)
        plt.close()
