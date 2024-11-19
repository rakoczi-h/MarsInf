from oct2py import Oct2Py
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from .layer import Layer
from .gravity import GravityMap
from .utils import multivariate, matern_covariance, great_circle_distance

plt.style.use('seaborn-v0_8')

class Planet():
    """
    Class for specific planet object with two layers, the crust and the mantle. It is designed for modelling the MOHO based on some parameters and to calculated a forward model to get a global gravity map/coefficients.
    Parameters
    ----------
        radius: float or int
            The radius of the planet in m. (Default: radius of Mars)
        mass: float or int
            The mass of the planet in kg. (Default: mass of Mars)
        crust: Layer object
            The object describing the upper layer of the planet. See layer.py. (Default: None)
        mantle: Layer object
            The object describing the lower layer of the planet. See layer.py. (Default: None)
        parameters: dict
            The parameters of the planet. Usually contains information about the aprameters of the density distribution of the crust and mantle. Each element of the dictionary is also separately saved as an attribute to the class. If provided at initialisation it overwrites every other default or input value for these attributes. (Default: None)
            If None, default values of these parameters are used:
                "av_dens_c": 3050.0 [kg/m^3], average density of crust
                "av_den_m": 3550.0 [kg/m^3], average density of mantle
                Parameter defining the matern covariance function for the density distribution of the two layers:
                "k_c": 0.6, kappa
                "k_m": 0.6
                "e_c": 10.0, epsilon
                "e_m": 10.0
                "v_c": 100.0 [kg/m^3], variance
                "v_m": 100.0 [kg/m^3]
        lat, long: np.ndarrays
            These are assumed to be created with meshgrid, hence these are 2D arrays with shape [no. of points in latitude grid, no. of points in longitude grid] and reversed, respectively. The shape parameter is inferred using lat. Units: degrees (Default: None)
        psi: np.ndarray
            The great circle distances between every single pair of points in the lat-long grid. The shape is [N x N], where N=no. of points in latitude x no.of points in longitude. (Default: None)
        seed: bool
            If True, all random processes will run with seed number 4. (Default: False)
        resolution: list
            The resolution in [latitude, longitude] in degrees. (Default: None)
        topography: np.ndarray
            A 1D array continaing the values of the surface topography. The size should be N, where N=no. of points in latitude x no.of points in longitude. Units: m. (Default: None)
        Te: float or int
            The effective elastic thickness of the lithospehre. Used when calculating the MOHO from the topography. Units: m (Default: 80000.0)
        D_c: float or int
            Crustal thickness. Used when calculating the MOHO from the topography. Units: m (Default: 60000.0)
        E: Young's modulus. Used when calculating the MOHO from the topography. Units: Pa (Default: 1e11)
        v: Poisson's ratio. Used when calculating the MOHO from the topography. Unitless. (Default: 0.25)
        seed: bool
            If true, seed number 4 is used.
    """
    def __init__(self, radius=3.396*1e6, mass=0.64171*1e24, crust=None, mantle=None, parameters=None, lat=None, long=None, psi=None, seed=False, resolution=None, topography=None, Te=80000.0, D_c=60000.0, E=1e11, v=0.25):
        self.radius = radius
        self.mass = mass
        self.crust = crust
        self.mantle = mantle
        self.lat = lat 
        self.long = long
        self.topography = topography
        self.shape = np.shape(lat) # has to be [lat, long]
        self.resolution = resolution
        self.psi = psi
        self.Te = Te #80-100 for Mars
        self.D_c = D_c #60+-10 for Mars
        self.E = E
        self.v = v
        self.parameter_labels = None
        self.parameters = parameters
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
                value.setdefault("type", "sh")
                value.setdefault("av_dens_c", 3050.0)
                value.setdefault("av_den_m", 3550.0)
                value.setdefault("k_c", 0.6)
                value.setdefault("k_m", 0.6)
                value.setdefault("e_c", 10.0)
                value.setdefault("e_m", 10.0)
                value.setdefault("v_c", 100.0)
                value.setdefault("v_m", 100.0)
        super().__setattr__(name,value)

    def make_planet(self, verbose=False):
        """
        Function to construct the whole model of the planet based on the provided parameters to the class.
        Parameters
        ----------
            verbose: bool
                If true, more updates are printed along with timings.
        """
        # Checking whether seeding or not
        seed = None

        if self.psi is None:
            self.psi = great_circle_distance(long=self.long.flatten()/180.0*np.pi, lat=self.lat.flatten()/180.0*np.pi)

        # Making the crust
        self.crust = Layer(parameters={'av_dens': self.av_dens_c, 'kappa': self.k_c,
                                'epsilon': self.e_c, 'var': self.v_c},
                        lat=self.lat,
                        long=self.long,
                        psi=self.psi)
        if self.seed:
            seed = 1
        self.crust.make_dens_model(seed=seed, verbose=verbose)
        self.crust.topo_model = self.topography
        if verbose:
            print("Made crust...")

        # Making the crust
        self.mantle = Layer(parameters={'av_dens': self.av_dens_m, 'kappa': self.k_m,
                                'epsilon': self.e_m, 'var': self.v_m},
                        lat=self.lat,
                        long=self.long,
                        psi=self.psi)
        if self.seed:
            seed = 3
        self.mantle.make_dens_model(seed=seed, verbose=verbose)

        if verbose:
            print("Made mantle...")

        # If the topography is not given, random topography is made with pre-set covariance
        if self.crust.topo_model is None:
            print('Generating random topography as self.crust.topo_model is None.')
            if self.seed:
                seed = 5
            self.crust.topo_model = self.random_topo(seed=seed, verbose=verbose)
            if verbose:
                print("Made topography...")

        # If not already given, then the moho is inferred from the topography
        if self.mantle.topo_model is None:
            self.mantle.topo_model = self.make_moho(verbose=verbose)
            if verbose:
                print("Made crust-mantle boundary...")
        print("Constructed planet...")

    def make_moho(self, moho_parameters=None, topography=None, verbose=False):
        """
        Function using the Thin Shell flexure model to infer the MOHO from the surface topography.
        Parameters
        ----------
            moho_parameters: dict
                Has to contain the keys: 'Te', 'D_c', 'E', 'v', 'rho_m', 'rho_c', 'GM', Re'
                If None, the parameters are taken from the class. 
            topography: np.ndarray
                A 1D array continaing the values of the surface topography. The size should be N, where N=no. of points in latitude x no.of points in longitude. Units: m. (Default: None)
                If None, taking it from the class.
            verbose: bool
        Output
        ------
            moho: np.ndarray
                A 1D array containing the values of the mantle-crust boundary topography. Units: m. The size is N, where N=no, of points in latitude x no. of points in longitude.
        """
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
        dir_path = os.path.dirname(os.path.realpath(__file__))
        octave.addpath(os.path.join(dir_path, 'gsh_tools/'))

        topography = np.reshape(topography, self.shape) #topography has to be 2D here
        moho = octave.topo2crust(topography, self.shape[0]-1, 'Thin_Shell', moho_parameters, nout=1)
        moho = - moho
        #moho = -self.D_c - moho
        octave.exit()
        if verbose:
            print(f"Time elapsed by make_moho: {datetime.now()-start}")
        return moho.flatten()

    def random_topo(self, parameters={'av_elev': 0.0, 'kappa': 0.6, 'epsilon': 10, 'var': 9000}, seed=None, verbose=False):
        """
        Generates random topography from parameters using matern covariance function and multivariate gaussian sampling.
        Parameters
        ----------
            parameters: dict
                Has to have keys: 'av_elev', 'kappa', 'epsilon', 'var. D(efault: {'av_elev': 0.0, 'kappa': 0.6, 'epsilon': 10, 'var': 9000})
            seed: int
                If given, this seed number is used for the multivaraite sampling. (Default: None)
            verbose: bool
        Output:
        ------
            topo: np.nd.array
                A 1D array continaing the values of the surface topography. The size is N, where N=no. of points in latitude x no.of points in longitude. Units: m. (Default: None)

        """
        matern = matern_covariance(self.psi, parameters['epsilon'], parameters['kappa'], parameters['var'], timed=verbose)

        topo = multivariate(matern, parameters['av_elev']*np.ones(np.shape(matern)[0]), seed=seed, timed=verbose)
        self.topography = topo
        return topo

    def forward_model(self, height=None, return_SH=False):
        """
        Function that computes either the SH coefficients or the map of global gravity from the planet model.
        Parameters:
        ----------
            height: float or int
                The height above the surface at which the gravity map is to be obtained. Not necessary when return_SH is True. Units: m (Default: None)
            return_SH: bool
                If True, only the SH coefficients are returned, if False, then the gravity maps are constructed also. (Default: False)
        Output:
        ------
            gravity: GravityMap
                Object containing information about the gravitational field. If return_SH is True, then only the SH coefficients are included in the class, no spatial gravity map.
                If return_SH is False, then the class also contains graviational acceleration and gradient information.
        """
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
        dir_path = os.path.dirname(os.path.realpath(__file__))
        octave.addpath(os.path.join(dir_path, 'gsh_tools/'))

        V = octave.model_SH_analysis(input_model, nout=1) # getting SH coefficients from model.
        if return_SH:
            gravity = GravityMap(lat=self.lat, long=self.long, height=height, resolution=self.resolution, coeffs=V)
            start_octave_close = datetime.now()
            octave.exit()
            return gravity # returning GravityMap with only coefficients
        else:
            if height is None:
                raise ValueError('Need to provide height to obtain gravity map.')

            latLim = [np.min(np.min(self.lat)), np.max(np.max(self.lat)), self.resolution[0]]
            lonLim = [np.min(np.min(self.long)), np.max(np.max(self.long)), self.resolution[1]]
            SHbounds = [0.0, self.shape[0]-1] # truncating at 2nd degree for normalisation
            data = octave.model_SH_synthesis(lonLim,latLim,height,SHbounds,V,input_model,nout=1) # getting the gravity fields
            data.vec.X = np.flip(data.vec.X) # flipping the data to align with topography
            data.ten.Tzz = np.flip(data.ten.Tzz)
            data.vec.Y = np.flip(data.vec.Y)
            data.vec.Z = np.flip(data.vec.Z)
            g = {'X': data.vec.X, 'Y': data.vec.Y, 'Z': data.vec.Z}
            grad = {'zz': data.ten.Tzz}
            gravity = GravityMap(g=g, grad=grad, lat=self.lat, long=self.long, height=height, coeffs=V, resolution=self.resolution)
            octave.exit()
            return gravity # returning Gravitymap with both coefficients and fields.

    # Plotting tools
    def plot_layers(self, filename='layers.png'):
        """
        Plots the topography on the surface, the MOHO, and the density distributions of the crust and mantle.
        filename: str
            The name under which the file is saved.
        """
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
