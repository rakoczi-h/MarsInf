from oct2py import Oct2Py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
from datetime import datetime

from .layer import Layer
from .gravity import GravityMap
from .utils import multivariate, matern_covariance, great_circle_distance

plt.style.use('seaborn-v0_8')

class Planet():
    def __init__(self, radius=3.396*1e6, mass=0.652*1e24, crust=None, mantle=None, parameters=None, model=None, lat=None, long=None, psi=None, shape=None, seed=False, resolution=None, topography=None):
        self.radius = radius
        self.mass = mass
        self.crust = crust
        self.mantle = mantle
        self.parameter_labels = None
        self.parameters = parameters
        self.lat = lat # assumed that it is in degrees
        self.long = long
        self.coordinates = None
        self.topography = topography
        if self.lat is not None and self.long is not None:
            self.coordinates = np.c_[self.lat.flatten(), self.long.flatten()]
        self.shape = shape # has to be [lat, long]
        self.resolution = resolution
        self.psi = psi
        self.model = model
        self.Te = 50000.0 #80-100
        self.D_c = 50000.0 #60+-10
        self.E = 100.0*1e9
        self.v = 0.25
        self.seed = seed

    def __setattr__(self, name, value):
        if name == 'crust':
            if value is not None:
                if not isinstance(value, Layer):
                    raise ValueError('The crust has to be a Layer object.')
        if name == 'mantle':
            if value is not None:
                if not isinstance(value, Layer):
                    raise ValueError('The mantle has to be a Layer object.')
        if name == 'parameters':
            if value is not None:
                if not isinstance(value, dict):
                    raise ValueError('parameters has to be a dictionary')
                if self.parameter_labels is None:
                    self.parameter_labels = list(value.keys())
                for i, key in enumerate(self.parameter_labels):
                    super().__setattr__(key, value[key])
        super().__setattr__(name,value)

    def make_planet(self, verbose=False):
        if self.seed:
            seed  = 4
        else:
            seed = None
        self.psi = great_circle_distance(long=self.long.flatten()/180.0*np.pi, lat=self.lat.flatten()/180.0*np.pi)
        self.crust = Layer(parameters={'av_dens': self.av_dens_c, 'kappa': self.k_c,
                                'epsilon': self.e_c, 'var': self.v_c},
                        lat=self.lat,
                        long=self.long,
                        psi=self.psi)
        self.crust.make_dens_model(seed=seed, verbose=verbose)
        self.crust.topo_model = self.topography
        if verbose:
            print("Made crust...")
        self.mantle = Layer(parameters={'av_dens': self.av_dens_m, 'kappa': self.k_m,
                                'epsilon': self.e_m, 'var': self.v_m},
                        lat=self.lat,
                        long=self.long,
                        psi=self.psi)
        self.mantle.make_dens_model(seed=seed, verbose=verbose)
        if verbose:
            print("Made mantle...")
        if self.crust.topo_model is None:
            print('Generating random topography as self.crust.topo_model is None.')
            self.crust.topo_model = self.random_topo(seed=seed, verbose=verbose)
            if verbose:
                print("Made topography...")
        if self.mantle.topo_model is None:
            self.mantle.topo_model = self.make_moho()
            if verbose:
                print("Made crust-mantle boundary...")
        print("Constructed planet...")

    def make_moho(self, moho_parameters=None, topography=None):
        start = datetime.now()
        if self.shape is None:
            raise ValueError('Please provide the shape of the angle grid to the method.')
        if topography is None:
            if self.crust.topo_model is None:
                raise ValueError('Topography model of the crust needs to be defined.')
            else:
                topography = self.crust.topo_model
        if moho_parameters is None:
            moho_parameters = {'Te': self.Te,
                                'D_c': self.D_c,
                                'E': self.E,
                                'v': self.v,
                                'rho_m': self.av_dens_m,
                                'rho_c': self.av_dens_c,
                                'GM': 6.6743*1e-11*self.mass,
                                'Re': self.radius}
        octave = Oct2Py()
        octave.addpath('/home/2263373r/mars/marsinf/gsh_tools/')
        topography = np.reshape(topography, self.shape)
        moho = octave.topo2crust(topography, self.shape[0]-1, 'Thin_Shell', moho_parameters, nout=1)
        moho = - moho
        #moho = -self.D_c - moho
        octave.exit()
        print(f"Time elapsed by make_moho: {datetime.now()-start}")
        return moho.flatten()

    def random_topo(self, parameters={'av_elev': 0.0, 'kappa': 0.6, 'epsilon': 10, 'var': 9000}, seed=None, verbose=False):
        matern = matern_covariance(self.psi, parameters['epsilon'], parameters['kappa'], parameters['var'], timed=verbose)

        topo = multivariate(matern, parameters['av_elev']*np.ones(np.shape(matern)[0]), seed=seed, timed=verbose)
        self.topography = topo
        return topo

    def forward_model(self, height, return_SH=False):
        start = datetime.now()
        input_model = {'number_of_layers': 2,
                        'GM': 6.6743*1e-11*self.mass,
                        # 'Re_analyse': self.radius,
                        'Re': self.radius,
                        'geoid': 'none',
                        'nmax': self.shape[0]-1,
                        # 'correct_depth': 0.0,
                        'l1': {'bound': np.reshape(self.crust.topo_model, self.shape),
                                'dens': np.reshape(self.crust.dens_model, self.shape)},
                        'l2': {'bound': np.reshape(self.mantle.topo_model, self.shape),
                                'dens': np.reshape(self.mantle.dens_model, self.shape)},
                        'l3': {'bound': np.zeros(self.shape)}}
        octave = Oct2Py()
        octave.addpath('/home/2263373r/mars/marsinf/gsh_tools/')
        V = octave.model_SH_analysis(input_model, nout=1)
        if return_SH:
            gravity = GravityMap(lat=self.lat, long=self.long, height=height, shape=self.shape, resolution=self.resolution, coeffs=V)
            start_octave_close = datetime.now()
            octave.exit()
            return gravity
        else:
            latLim = [np.min(np.min(self.lat)), np.max(np.max(self.lat)), self.resolution[0]]
            lonLim = [np.min(np.min(self.long)), np.max(np.max(self.long)), self.resolution[1]]
            SHbounds = [2.0, self.shape[0]-1] # truncating at 2nd degree for normalisation
            data = octave.model_SH_synthesis(lonLim,latLim,height,SHbounds,V,input_model,nout=1)
            data.vec.X = np.flip(data.vec.X)
            data.ten.Tzz = np.flip(data.ten.Tzz)
            data.vec.Y = np.flip(data.vec.Y)
            data.vec.Z = np.flip(data.vec.Z)
            g = {'X': data.vec.X, 'Y': data.vec.Y, 'Z': data.vec.Z}
            grad = {'zz': data.ten.Tzz}
            gravity = GravityMap(g=g, grad=grad, lat=self.lat, long=self.long, height=height, shape=self.shape, coeffs=V, resolution=self.resolution)
            octave.exit()
            return gravity

    # Plotting tools
    def plot_layers(self, filename='layers.png'):
        plot_arrays = [self.crust.topo_model, self.crust.dens_model, self.mantle.topo_model, self.mantle.dens_model]
        titles = ['Topography', 'Crust', 'Moho', 'Mantle']
        labels = ['height [m]', r'$\rho$ [kg/m$^3$]', 'height [m]', r'$\rho$ [kg/m$^3$]']
        proj = 'mollweide'
        fig, axes = plt.subplots(2,2, subplot_kw={'projection': proj})
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            to_plot = np.reshape(plot_arrays[i], self.shape)

            im = ax.pcolormesh(self.long*np.pi/180.0, self.lat*np.pi/180.0, to_plot, cmap='rainbow')
            ax.set_title(titles[i])
            axpos = ax.get_position()
            pos_x = axpos.x0
            pos_y = axpos.y0 - axpos.height/2 + 0.1
            cax_width = axpos.width
            cax_height = 0.01
            pos_cax = fig.add_axes([pos_x,pos_y,cax_width,cax_height])
            plt.colorbar(im, cax=pos_cax, orientation='horizontal', label=labels[i])
        plt.savefig(filename)
        plt.close()
