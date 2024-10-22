import numpy as np
from datetime import datetime

from .utils import make_long_lat, great_circle_distance, matern_covariance, multivariate
from .planet import Planet
from .layer import Layer
from .prior import Prior

class PlanetDataSet():
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
            value.setdefault("av_dens_c", 3050.0)
            value.setdefault("av_den_m", 3550.0)
            value.setdefault("mass", 0.652*1e24)
            value.setdefault("radius", 3.396*1e6)
            value.setdefault("flex_model", 'Thin_Shell')
            value.setdefault("moho_parameters", {'Te': 50000.0, 'D_c': 50000.0, 'E': 100.0*1e9, 'v': 0.25})
        if name == 'survey_framework':
            if not isinstance(value, dict):
                raise ValueError("Expected dict for survey_framework.")
            value.setdefault("noise_scale", 0.0)
            value.setdefault("ranges", [[-180.0,180.0],[-90.0, 90.0],[0]])
            value.setdefault("resolution", [2,2])
            value.setdefault("height", 0.0)
        super().__setattr__(name, value)

    def make_dataset(self, parameters_dict=None):
        start_dataset = datetime.now()
        # Sampling the prior
        if parameters_dict is None:
            parameters_dict = self.priors.sample(size=self.size, returntype='dict')
            dataset_size = self.size
        else:
            if self.parameter_labels != list(parameters_dict.keys()):
                raise ValueError("The prior and the parameter keys do not match")
            dataset_size = np.shape(parameters_dict[list(parameters_dict.keys())[0]])[0]

        # Making the latitude longitude meshgrid
        Long, Lat = make_long_lat(self.survey_framework['resolution'], self.survey_framework['ranges'])
        # Calculating the array of great circle distances
        psi = great_circle_distance(long=Long.flatten()/180.0*np.pi, lat=Lat.flatten()/180.0*np.pi)

        # Simulating topography if not given
        if self.topography is None:
            print("Making topography...")
            cm_t = matern_covariance(psi, 10, 0.6, 1000)
            self.topography = multivariate(cm_t, 0.0*np.ones(np.shape(cm_t)[0]), seed=5)

        # Simulating the MOHO from self.topography
        moho_planet = Planet(lat=Lat, long=Long, shape=np.shape(Lat), resolution=self.survey_framework['resolution'])
        moho_parameters = self.model_framework['moho_parameters'] | {'rho_m': self.model_framework['av_dens_m'], 'rho_c': self.model_framework['av_dens_c'], 'GM': self.model_framework['mass']*6.6743*1e-11, 'Re': self.model_framework['radius']}
        moho = moho_planet.make_moho(moho_parameters, topography=self.topography)

        start_planets = datetime.now()
        # Maing planets
        planets = []
        for s in range(self.size):
            # picking the specific instance of planet parameters
            planet_parameters = dict.fromkeys(self.priors.keys)
            for key in list(self.parameter_labels):
                planet_parameters[key] = parameters_dict[key][s]
            crust = Layer(parameters={'av_dens': self.model_framework['av_dens_c'], 'kappa': planet_parameters['k_c'], 'epsilon': planet_parameters['e_c'], 'var': planet_parameters['v_c']}, lat=Lat, long=Long, psi=psi)
            crust.make_dens_model(seed=None)
            crust.topo_model = self.topography
            mantle = Layer(parameters={'av_dens': self.model_framework['av_dens_m'], 'kappa': planet_parameters['k_m'], 'epsilon': planet_parameters['e_m'], 'var': planet_parameters['v_m']}, lat=Lat, long=Long, psi=psi)
            mantle.make_dens_model(seed=None)
            mantle.topo_model = moho
            planet = Planet(parameters=planet_parameters, lat=Lat, long=Long, shape=np.shape(Lat), resolution=self.survey_framework['resolution'], psi=psi, crust=crust, mantle=mantle, mass=self.model_framework['mass'], radius=self.model_framework['radius'])
            planets.append(planet)
            if s % 1000 == 0:
                print(f"{s}/{self.size} data planets made. \t Time taken: {datetime.now()-start_planets}")
        self.planets = planets

        start_gravity_maps = datetime.now()
        gravitymaps = []
        for i, p in enumerate(planets):
            gravitymap = p.forward_model(height=self.survey_framework['height'])
            # can make faster by only doing Legendre_functions once?
            gravitymaps.append(gravitymap)
            if i % 1000 == 0:
                print(f"{i}/{self.size} data gravity maps made. \t Time taken: {datetime.now()-start_gravity_maps}")
        self.gravitymaps = gravitymaps

        print(f"Dataset made. Time elapsed: {datetime.now()-start_dataset}")
        return planets, gravitymaps

