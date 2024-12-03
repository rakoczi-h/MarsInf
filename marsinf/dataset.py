import numpy as np
from datetime import datetime
from oct2py import Oct2Py
import os

from .utils import make_long_lat, great_circle_distance, matern_covariance, multivariate
from .planet import Planet
from .layer import Layer
from .prior import Prior
from .gravity import GravityMap

class PlanetDataSet():
    """
    Parameters:
    ----------
        size: int
            The number of planets included in the data set.
        priors: Prior
            The object describing the prior probability distribution of the parameters defining the planets in the dataset.
        survey_framework: dict
            The dictionary containing some essential information about the gravity maps that correspond to the planets.
            If an empty dictionary is passed to the class, then the following default values are set:
                "ranges": [[-180.0,180.0],[-90.0, 90.0]], The ranges covered by the gravity map in [longitude, latitude]. Assumed to be in degrees.
                "resolution": [2,2], The resolution of the latitude and longitude grids in degrees.
                "height": 0.0, The height above the aeroid in m, at which the gravity measurement were taken.
        model_framework: dict
            The dictionary containing some essential information about the planet models that are contained within this dataset.
            These parameters are general and apply to all planets in the set. If an empty dictionary is passed to the class, then the following default values are set:
                "type": "sh", the type of gravity model to be constructed. Can be 'sh' or 'map'
                "av_dens_c": 3050.0, kg/m^3, the average density of the crust
                "av_den_m": 3550.0, kg/m^3, the average density of the mantle
                "mass": 0.652*1e24, kg, the mass of the planet. Defaults to Mars
                "radius": 3.396*1e6, m, the radius of the planet. Defaults to Mars
                "flex_model": 'Thin_Shell', the type of flexure model that is used to construct the MOHO. Only 'Thin_Shell' is accepted at the moment.
                "moho_parameters": {'Te': 80000.0, 'D_c': 60000.0, 'E': 100.0*1e9, 'v': 0.25}, The parameters that are used to infer the MOHO from topography.
                    See more detailed description in the Planet class.
        topography: np.ndarray
             1D array containing the topography elevation values. Has shape N. Units: m. (Default: None).
    """
    def __init__(self, size: int, priors: Prior, survey_framework={}, model_framework={}, topography=None):
        self.size = size
        self.priors = priors
        self.survey_framework = survey_framework
        self.model_framework = model_framework
        self.parameter_labels = priors.keys
        self.topography = topography
        self.planets = None
        self.gravitymaps = None

    def __setattr__(self, name, value):
        if name == 'model_framework':
            if not isinstance(value, dict):
                raise ValueError("Expected dict for model_framework.")
            value.setdefault("type", "sh")
            value.setdefault("av_dens_c", 3050.0)
            value.setdefault("av_den_m", 3550.0)
            value.setdefault("mass", 0.652*1e24)
            value.setdefault("radius", 3.396*1e6)
            value.setdefault("flex_model", 'Thin_Shell')
            value.setdefault("moho_parameters", {'Te': 80000.0, 'D_c': 60000.0, 'E': 100.0*1e9, 'v': 0.25})
        if name == 'survey_framework':
            if not isinstance(value, dict):
                raise ValueError("Expected dict for survey_framework.")
            value.setdefault("ranges", [[-180.0,180.0],[-90.0, 90.0]])
            value.setdefault("resolution", [2,2])
            value.setdefault("height", 0.0)
        super().__setattr__(name, value)

    def make_dataset(self, parameters_dict=None, slim_output=False, repeats=1):
        """
        Method that constructs the dataset and creates the self.gravitymaps and self.planets attributes.
        Parameters:
        ----------
            parameters_dict: dict
                The keys of the dictionary are the different parameters that are used to construct the data set.
                Each member of the dictionary has to be a 1D array, and these have to have the same length.
                If None, the priors are sampled for the parameters.
                Recognised parameters: 'e_c', 'k_c', 'v_c', 'e_m', 'k_m', 'v_m'
            slim_output: bool
                Decides whether to create smaller output.
                If True: The output is a dictionary, with the keys: 'gravity', 'sh_degrees', 'lat', 'long', 'shape', 'resolution'
                If False: The outputs are lists of classes of Planets and GravityMaps.
            repeats: int
                The number of times the same set of parameters are reused during the modelling.
                Since these are used to create a covariance matrix, which is then used to sample from a multivariate distribution,
                    the resulting density distributions will still differ.
        Output:
        ------
             Defined by slim_output.
        """
        start_dataset = datetime.now()
        # Sampling the prior
        if self.size % repeats != 0:
            dataset_size_new = round(self.size, -1)
            if dataset_size_new < self.size:
                dataset_size_new = dataset_size_new + repeats
            self.size = dataset_size_new
            print(f"Rounding data set size to {dataset_size}")

        if parameters_dict is None:
            parameters_dict = self.priors.sample(size=int(self.size/repeats), returntype='dict')
        else:
            if self.parameter_labels != list(parameters_dict.keys()):
                raise ValueError("The prior and the parameter keys do not match")
            if self.size != np.shape(parameters_dict[list(parameters_dict.keys())[0]])[0]:
                raise ValueError("Number of elements parameters_dict do not agree with the dataset size")

        # Making the latitude longitude meshgrid
        Long, Lat = make_long_lat(self.survey_framework['resolution'], self.survey_framework['ranges'])
        # Calculating the array of great circle distances
        psi = great_circle_distance(long=Long.flatten()/180.0*np.pi, lat=Lat.flatten()/180.0*np.pi)


        # Simulating topography if not given
        if self.topography is None:
            print("Making topography...")
            cm_t = matern_covariance(psi, 10.0, 0.6, 100.0)
            self.topography = multivariate(cm_t, 0.0*np.ones(np.shape(cm_t)[0]), seed=self.model_framework['seed_topography'])

        # Simulating the MOHO from self.topography
        moho_planet = Planet(lat=Lat, long=Long, resolution=self.survey_framework['resolution'])
        moho_parameters = self.model_framework['moho_parameters'] | {'rho_m': self.model_framework['av_dens_m'],
                                                                        'rho_c': self.model_framework['av_dens_c'],
                                                                        'GM': self.model_framework['mass']*6.6743*1e-11,
                                                                        'Re': self.model_framework['radius']}
        moho = moho_planet.make_moho(moho_parameters, topography=self.topography)

        start_planets = datetime.now()
        # Making planets
        planets = []
        for i in range(int(self.size/repeats)):
            # picking the specific instance of planet parameters
            planet_parameters = dict.fromkeys(self.priors.keys)
            for key in list(self.parameter_labels):
                planet_parameters[key] = parameters_dict[key][i]

            # make the matern covariances
            crust_matern = matern_covariance(psi=psi, epsilon=planet_parameters['e_c'],
                                            kappa=planet_parameters['k_c'], sigma=planet_parameters['v_c'])
            mantle_matern = matern_covariance(psi=psi, epsilon=planet_parameters['e_m'],
                                            kappa=planet_parameters['k_m'], sigma=planet_parameters['v_m'])

            for j in range(repeats):
                mantle = Layer(parameters={'av_dens': self.model_framework['av_dens_m'],
                                            'kappa': planet_parameters['k_m'],
                                            'epsilon': planet_parameters['e_m'],
                                            'var': planet_parameters['v_m']})
                mantle.matern = mantle_matern.copy()
                mantle.topo_model = moho.copy()

                crust = Layer(parameters={'av_dens': self.model_framework['av_dens_c'],
                                            'kappa': planet_parameters['k_c'],
                                            'epsilon': planet_parameters['e_c'],
                                            'var': planet_parameters['v_c']})
                crust.matern = crust_matern.copy()
                crust.topo_model = self.topography.copy()

                crust.make_dens_model(seed=self.model_framework['seed_crust'])
                mantle.make_dens_model(seed=self.model_framework['seed_mantle'])
                crust.matern = None
                mantle.matern = None

                planet = Planet(parameters=planet_parameters, lat=Lat, long=Long,
                                resolution=self.survey_framework['resolution'], psi=psi, crust=crust,
                                mantle=mantle, mass=self.model_framework['mass'], radius=self.model_framework['radius'])
                planets.append(planet)
                if (i+1)*(j+1) % 100 == 0:
                    print(f"{i*j}/{self.size} planets made. \t Time taken: {datetime.now()-start_planets}")
        self.planets = planets

        start_gravity_maps = datetime.now()
        if self.model_framework['type'] == 'sh':
            gravitymaps = self.forward_model(return_SH=True, slim_output=slim_output)
        elif self.model_framework['type'] == 'map':
            gravitymaps = self.forward_model(return_SH=False, slim_output=slim_output)
        print(f"Dataset made. Time elapsed: {datetime.now()-start_dataset}")
        self.gravitymaps = gravitymaps

        if slim_output:
            for key in parameters_dict.keys():
                # repeating the same parameters 10 times since the matern was reused
                parameters_dict[key] = np.repeat(parameters_dict[key], repeats)
            if self.model_framework['type'] == 'sh':
                return parameters_dict | {'gravity': [g[:,2:] for g in gravitymaps],
                                      'sh_degrees': gravitymaps[0][:,:2],
                                      'lat': self.planets[0].lat,
                                      'long': self.planets[0].long,
                                      'shape': self.planets[0].shape,
                                      'resolution': self.planets[0].resolution}
            elif self.model_framework['type'] == 'map':
                return parameters_dict | {'gravity_coeffs': [g[:,2:] for g in gravitymaps.coeffs],
                                        'gravity_acceleration': [g for g in gravitymaps.g['Z']],
                                      'sh_degrees': gravitymaps[0].coeffs[:,:2],
                                      'lat': self.planets[0].lat,
                                      'long': self.planets[0].long,
                                      'shape': self.planets[0].shape,
                                      'resolution': self.planets[0].resolution}
        else:
            return planets, gravitymaps

    def forward_model(self, return_SH=False, slim_output=True):
        """
        Computes the gravity from the planet parameters. Functions the same as the method with the same name in the Planet class,
        but only opens octave once, which provides significant speed-up.
        Parameters
        ----------
            return_SH: bool
                If True, only the SH coefficients are returned, if False, then the gravity maps are constructed also. (Default: False)
            slim_output: bool
                If True and return_SH is True: a list of SH coefficients of the planets
                        and return_SH is False: a list of the GravityMaps
                if False: a list of the GravityMaps
        Output:
        ------
             Defined by slim_output.
        """
        start = datetime.now()
        octave = Oct2Py()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        octave.addpath(os.path.join(dir_path, 'gsh_tools/'))
        height = self.survey_framework['height']
        gravitymaps = []
        for i, p in enumerate(self.planets):
            input_model = {'number_of_layers': 2,
                            'GM': 6.6743*1e-11*p.mass,
                            # 'Re_analyse': self.radius,
                            'Re': p.radius,
                            'geoid': 'none',
                            'nmax': p.shape[0]-1,
                            # 'correct_depth': 0.0,
                            'l1': {'bound': np.reshape(p.crust.topo_model, p.shape),
                                    'dens': np.reshape(p.crust.dens_model, p.shape)},
                            'l2': {'bound': np.reshape(p.mantle.topo_model, p.shape),
                                    'dens': np.reshape(p.mantle.dens_model, p.shape)},
                            'l3': {'bound': np.zeros(p.shape)}}
            V = octave.model_SH_analysis(input_model, nout=1)
            if return_SH:
                gravity = GravityMap(lat=p.lat, long=p.long,
                                    height=height, resolution=p.resolution, coeffs=V)
            else:
                latLim = [np.min(np.min(p.lat)), np.max(np.max(p.lat)), p.resolution[0]]
                lonLim = [np.min(np.min(p.long)), np.max(np.max(p.long)), p.resolution[1]]
                SHbounds = [0.0, p.shape[0]-1]
                data = octave.model_SH_synthesis(lonLim,latLim,height,SHbounds,V,input_model,nout=1)
                data.vec.X = np.flip(data.vec.X)
                data.ten.Tzz = np.flip(data.ten.Tzz)
                data.vec.Y = np.flip(data.vec.Y)
                data.vec.Z = np.flip(data.vec.Z)
                g = {'X': data.vec.X, 'Y': data.vec.Y, 'Z': data.vec.Z}
                grad = {'zz': data.ten.Tzz}
                gravity = GravityMap(g=g, grad=grad, lat=p.lat, long=p.long,
                                    height=height, coeffs=V, resolution=p.resolution)
            gravitymaps.append(gravity)
            if i+1 % 100 == 0:
                print(f"{i+1}/{self.size} gravity maps made. \t Time taken: {datetime.now()-start}")
        octave.exit()
        if slim_output:
            if return_SH:
                return [g.coeffs for g in gravitymaps]
            else:
                return gravitymaps
        else:
            return gravitymaps


