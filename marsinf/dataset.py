import numpy as np
from datetime import datetime
from oct2py import Oct2Py
import resource
import os

from .utils import make_long_lat, great_circle_distance, matern_covariance, multivariate
from .planet import Planet
from .layer import Layer
from .prior import Prior
from .gravity import GravityMap

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
            value.setdefault("type", "sh")
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

    def make_dataset(self, parameters_dict=None, slim_output=True, repeats=10):
        start_dataset = datetime.now()
        # Sampling the prior
        if self.size % repeats != 0:
            dataset_size_new = round(self.size, -1)
            if dataset_size_new < self.size:
                dataset_size_new = dataset_size_new + repeats
            self.size = dataset_size_new
            print(f"Rounding data set size to {dataset_size}")

        if parameters_dict is None:
            parameters_dict = self.priors.sample(size=int(self.size/10), returntype='dict')
        else:
            if self.parameter_labels != list(parameters_dict.keys()):
                raise ValueError("The prior and the parameter keys do not match")
            if self.size != np.shape(parameters_dict[list(parameters_dict.keys())[0]])[0]:
                raise ValueError("Number of elements parameters_dict do not agree with the dataset size")

        # Making the latitude longitude meshgrid
        Long, Lat = make_long_lat(self.survey_framework['resolution'], self.survey_framework['ranges'])
        # Calculating the array of great circle distances
        psi = great_circle_distance(long=Long.flatten()/180.0*np.pi, lat=Lat.flatten()/180.0*np.pi)
        print(np.shape(psi))

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
        # Making planets
        planets = []
        for i in range(int(self.size/repeats)):
            # picking the specific instance of planet parameters
            planet_parameters = dict.fromkeys(self.priors.keys)
            for key in list(self.parameter_labels):
                planet_parameters[key] = parameters_dict[key][i]

            # make the matern covariances
            crust_matern = matern_covariance(psi=psi, epsilon=planet_parameters['e_c'], kappa=planet_parameters['k_c'], var=planet_parameters['v_c'])
            mantle_matern = matern_covariance(psi=psi, epsilon=planet_parameters['e_m'], kappa=planet_parameters['k_m'], var=planet_parameters['v_m'])

            for j in range(repeats):
                mantle = Layer(parameters={'av_dens': self.model_framework['av_dens_m'], 'kappa': planet_parameters['k_m'], 'epsilon': planet_parameters['e_m'], 'var': planet_parameters['v_m']})
                mantle.matern = mantle_matern.copy()
                mantle.topo_model = moho.copy()

                crust = Layer(parameters={'av_dens': self.model_framework['av_dens_c'], 'kappa': planet_parameters['k_c'], 'epsilon': planet_parameters['e_c'], 'var': planet_parameters['v_c']})
                crust.matern = crust_matern.copy()
                crust.topo_model = self.topography.copy()

                crust.make_dens_model(seed=None)
                mantle.make_dens_model(seed=None)
                crust.matern = None
                mantle.matern = None

                planet = Planet(parameters=planet_parameters, lat=Lat, long=Long, shape=np.shape(Lat), resolution=self.survey_framework['resolution'], psi=psi, crust=crust, mantle=mantle, mass=self.model_framework['mass'], radius=self.model_framework['radius'])
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
                parameters_dict[key] = np.repeat(parameters_dict[key], repeats) # repeating the same parameters 10 times since the matern was reused
            return parameters_dict | {'gravity': [g[:,2:] for g in gravitymaps], 'sh_degrees': gravitymaps[0][:,:2], 'lat': self.planets[0].lat, 'long': self.planets[0].long, 'shape': self.planets[0].shape, 'resolution': self.planets[0].resolution}
        else:
            return planets, gravitymaps


    def forward_model(self, return_SH=False, slim_output=True):
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
                gravity = GravityMap(lat=p.lat, long=p.long, height=height, shape=p.shape, resolution=p.resolution, coeffs=V)
            else:
                latLim = [np.min(np.min(p.lat)), np.max(np.max(p.lat)), p.resolution[0]]
                lonLim = [np.min(np.min(p.long)), np.max(np.max(p.long)), p.resolution[1]]
                SHbounds = [2.0, p.shape[0]-1] # truncating at 2nd degree for normalisation
                data = octave.model_SH_synthesis(lonLim,latLim,height,SHbounds,V,input_model,nout=1)
                data.vec.X = np.flip(data.vec.X)
                data.ten.Tzz = np.flip(data.ten.Tzz)
                data.vec.Y = np.flip(data.vec.Y)
                data.vec.Z = np.flip(data.vec.Z)
                g = {'X': data.vec.X, 'Y': data.vec.Y, 'Z': data.vec.Z}
                grad = {'zz': data.ten.Tzz}
                gravity = GravityMap(g=g, grad=grad, lat=p.lat, long=p.long, height=height, shape=p.shape, coeffs=V, resolution=p.resolution)
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


