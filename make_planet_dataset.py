import pickle as pkl
import os
import numpy as np
import pandas as pd
import sys
import resource
from datetime import datetime

from marsinf.prior import Prior
from marsinf.dataset import PlanetDataSet
from marsinf.utils import great_circle_distance, matern_covariance, multivariate, make_long_lat, sections_mean

n = int(sys.argv[1])

saveloc = '/scratch/balta0/2263373r/mars/two_layer_real_topo/'

if not os.path.exists(saveloc):
    os.mkdir(saveloc)

distributions = {'e_c': ['Uniform', 0.1, 20.0],
                'k_c': ['Uniform', 0.1, 1.5],
                'v_c': ['Uniform', 10.0, 500.0], #kg/m^3
                'e_m': ['Uniform', 0.1, 20.0],
                'k_m': ['Uniform', 0.1, 1.5],
                'v_m': ['Uniform', 10.0, 500.0] #kg/m^3
                }
priors = Prior(distributions=distributions)

with open(os.path.join(saveloc, 'priors.pkl'), 'wb') as file:
    pkl.dump(priors, file)

survey_framework = {'noise_scale': 0.0, #microGal?
                    'resolution': [4.0, 4.0], #deg
                    'ranges': [[-180.0, 180.0], [-90.0, 90.0]], #deg
                    'height': 0.0 #m
                    }

model_framework =  {'type': 'sh',
                    'av_dens_c': 3050.0, #kg/m^3
                    'av_dens_m': 3550.0, #kg/m^3
                    'flex_model': 'Thin_Shell', #'Thin_Shell', 'Airy', 'Infinite_Plate'
                    'moho_parameters': {'Te': 80000.0, #m
                                        'D_c': 60000.0, #m
                                        'E': 100.0*1e9, #young's modulus, Pa
                                        'v': 0.25, #poisson's ratio
                                        'GM': 4.283*1e13, #Nm^2/kg
                                        'Re': 3.396*1e6 #m
                                        },
                    'mass': 6.4171*1e23, #kg
                    'radius': 3.396*1e6 #m
                    }


topography = None
df = pd.read_csv('/scratch/balta0/2263373r/mars/megt90n000eb.csv', header=None)

shape = [int((survey_framework['ranges'][1][1]-survey_framework['ranges'][1][0])/survey_framework['resolution'][1]), int((survey_framework['ranges'][0][1]-survey_framework['ranges'][0][0])/survey_framework['resolution'][0])]

topography = df.to_numpy()
topography = np.c_[topography[:,int(np.shape(topography)[1]/2):], topography[:,:int(np.shape(topography)[1]/2)]]
topography = sections_mean(topography, shape)
topography = np.flip(topography, axis=0)
## ------------- MAKE MATERN COVARIANCES -----------------------
#long, lat = make_long_lat(resolution = survey_framework['resolution'],
#                            ranges = survey_framework['ranges'])
#
#psi = great_circle_distance(long.flatten()/180.0*np.pi, lat.flatten()/180.0*np.pi)
#
#size = 10
#parameters = priors.sample(size=size, returntype='dict')
#
## better to make 2 for loops so less data is saved at once
#materns = []
#for i in range(size):
#    matern_c = matern_covariance(psi=psi,
#                                epsilon = parameters['e_c'][i],
#                                kappa = parameters['k_c'][i],
#                                var = parameters['v_c'][i])
#    materns.append(matern_c)
#materns = np.array(materns)
#crust_dict = {'e_c': parameters['e_c'], 'k_c': parameters['k_c'], 'v_c': parameters['v_c'],
#                'matern': materns}
#with open(os.path.join(saveloc, f"matern_crust_{n}.pkl"), 'wb') as file:
#    pkl.dump(crust_dict, file)
#
#materns = []
#for i in range(size):
#    matern_m = matern_covariance(psi=psi,
#                                epsilon = parameters['e_m'][i],
#                                kappa = parameters['k_m'][i],
#                                var = parameters['v_m'][i])
#    materns.append(matern_m)
#materns = np.array(materns)
#crust_dict = {'e_m': parameters['e_m'], 'k_m': parameters['k_m'], 'v_m': parameters['v_m'],
#                'matern': materns}
#with open(os.path.join(saveloc, f"matern_mantle_{n}.pkl"), 'wb') as file:
#    pkl.dump(crust_dict, file)
#
# ------------------ MAKING DATASET FROM MATERNS ---------------------
#with open(os.path.join(saveloc, f"matern_mantle_{n}.pkl"), 'rb') as file:
#    matern_mantle = pkl.load(file)
#with open(os.path.join(saveloc, f"matern_crust_{n}.pkl"), 'rb') as file:
#    matern_crust = pkl.load(file)
#
#size = 10
#dt_train = PlanetDataSet(priors=priors, size=size, survey_framework=survey_framework, model_framework=model_framework)
#dt_train.make_dataset_from_matern(matern_crust, matern_mantle)
#file_name = os.path.join(saveloc, f"trainset_{n}.pkl")
#with open(file_name, 'wb') as file:
#    pkl.dump(dt_train, file)
#print(f"Data set of size {dt_train.size} made and saved as {file_name}.")
#

# ------------------ MAKING  DATASET FROM SCRATCH ---------------------
## Make training data
#size = 10000
#dt_train = PlanetDataSet(priors=priors, size=size, survey_framework=survey_framework, model_framework=model_framework, topography=topography)
#dt_train = dt_train.make_dataset(slim_output=True, repeats=5)
##dt_train.make_dataset(slim_output=False)
#file_name = os.path.join(saveloc, f"trainset_{n}.pkl")
#start_save = datetime.now()
#with open(file_name, 'wb') as file:
#    pkl.dump(dt_train, file)
#print(f"Data set of size {size} made and saved as {file_name}.")

## Make validation data
#size = 10000
#dt_train = PlanetDataSet(priors=priors, size=size, survey_framework=survey_framework, model_framework=model_framework, topography=topography)
#dt_train = dt_train.make_dataset(slim_output=True, repeats=5)
##dt_train.make_dataset(slim_output=False)
#file_name = os.path.join(saveloc, f"validationset_{n}.pkl")
#start_save = datetime.now()
#with open(file_name, 'wb') as file:
#    pkl.dump(dt_train, file)
#print(f"Data set of size {size} made and saved as {file_name}.")


## Make pp data
#size = 100
#dt_train = PlanetDataSet(priors=priors, size=size, survey_framework=survey_framework, model_framework=model_framework, topography=topography)
#dt_train = dt_train.make_dataset(slim_output=True, repeats=1)
##dt_train.make_dataset(slim_output=False)
#file_name = os.path.join(saveloc, f"ppset_{n}.pkl")
#start_save = datetime.now()
#with open(file_name, 'wb') as file:
#    pkl.dump(dt_train, file)
#print(f"Data set of size {size} made and saved as {file_name}.")

## Make test data
size = 10
dt_train = PlanetDataSet(priors=priors, size=size, survey_framework=survey_framework, model_framework=model_framework, topography=topography)
#dt_train = dt_train.make_dataset(slim_output=True, repeats=1)
dt_train.make_dataset(slim_output=False)
file_name = os.path.join(saveloc, f"example_planets.pkl")
start_save = datetime.now()
with open(file_name, 'wb') as file:
    pkl.dump(dt_train, file)
print(f"Data set of size {size} made and saved as {file_name}.")


# ---------------- MAKING DATASET WITH PRESET PARAMETERS ----------------------
#size=15
#dt_train = PlanetDataSet(priors=priors, size=size, survey_framework=survey_framework, model_framework=model_framework, topography=topography)
#parameters_dict = dict.fromkeys(priors.keys)
#parameters_dict['e_c'] = np.array([1.0, 3.0, 5.0, 10.0, 20.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
#parameters_dict['k_c'] = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.1, 0.3, 0.6, 0.9, 1.2, 0.6, 0.6, 0.6, 0.6, 0.6])
#parameters_dict['v_c'] = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 10.0, 50.0, 100.0, 200.0, 500.0])
#parameters_dict['e_m'] = 10.0*np.ones(size)
#parameters_dict['k_m'] = 0.6*np.ones(size)
#parameters_dict['v_m'] = 100.0*np.ones(size)
#dt_train = dt_train.make_dataset(slim_output=True, repeats=1, parameters_dict=parameters_dict)
#file_name = os.path.join(saveloc, f"testset_preset_{n}.pkl")
#start_save = datetime.now()
#with open(file_name, 'wb') as file:
#    pkl.dump(dt_train, file)
#print(f"Data set of size {size} made and saved as {file_name}.")

#size=10
#dt_train = PlanetDataSet(priors=priors, size=size, survey_framework=survey_framework, model_framework=model_framework, topography=topography)
#parameters_dict = dict.fromkeys(priors.keys)
#
#parameters_dict['e_c'] = 10.0*np.ones(size)
#parameters_dict['k_c'] = 0.6*np.ones(size)
#parameters_dict['v_c'] = 100.0*np.ones(size)
#parameters_dict['e_m'] = 10.0*np.ones(size)
#parameters_dict['k_m'] = 0.6*np.ones(size)
#parameters_dict['v_m'] = 100.0*np.ones(size)
#
#dt_train = dt_train.make_dataset(slim_output=True, repeats=10, parameters_dict=parameters_dict)
#file_name = os.path.join(saveloc, f"testset_preset_{n}.pkl")
#start_save = datetime.now()
#with open(file_name, 'wb') as file:
#    pkl.dump(dt_train, file)
#print(f"Data set of size {size} made and saved as {file_name}.")
#

