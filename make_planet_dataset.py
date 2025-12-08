import pickle as pkl
import os
import numpy as np
import pandas as pd
import sys
import resource
import json
from datetime import datetime

from marsinf.prior import Prior
from marsinf.dataset import PlanetDataSet
from marsinf.utils import great_circle_distance, matern_covariance, multivariate, make_long_lat, sections_mean


saveloc = 'data'
if not os.path.exists(saveloc):
    os.mkdir(saveloc)


# Defining priors
distributions = {'e_c': ['Uniform', 0.1, 20.0],
                'k_c': ['Uniform', 0.1, 1.5],
                'v_c': ['Uniform', 10.0, 500.0], #kg/m^3
                'e_m': ['Uniform', 0.1, 20.0],
                'k_m': ['Uniform', 0.1, 1.5],
                'v_m': ['Uniform', 10.0, 500.0] #kg/m^3
                }
priors = Prior(distributions=distributions)


# Defining the configuration of measurement data
survey_framework = {'noise_scale': 0.0, #microGal
                    'resolution': [4.0, 4.0], #deg
                    'ranges': [[-180.0, 180.0], [-90.0, 90.0]], #deg
                    'height': 0.0 #m
                    }

# Defining the configuration of the planetary models
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
                    'radius': 3.396*1e6, #m
                    'seed_topography': 123,
                    'seed_mantle': None,
                    'seed_crust': None
                    }

with open(os.path.join(saveloc, 'priors.pkl'), 'wb') as file:
    pkl.dump(priors, file)
with open(os.path.join(saveloc, 'survey_framework.json'), 'w') as file:
    json.dump(survey_framework, file)
with open(os.path.join(saveloc, 'model_framework.json'), 'w') as file:
    json.dump(model_framework, file)


# Either reading topography from file, or setting it to None.
# If None, it is randomly generated using the Matérn covariance function
#topography = None
df = pd.read_csv('megt90n000eb.csv', header=None)

shape = [int((survey_framework['ranges'][1][1]-survey_framework['ranges'][1][0])/survey_framework['resolution'][1]), int((survey_framework['ranges'][0][1]-survey_framework['ranges'][0][0])/survey_framework['resolution'][0])]

topography = df.to_numpy()
topography = np.c_[topography[:,int(np.shape(topography)[1]/2):], topography[:,:int(np.shape(topography)[1]/2)]]
topography = sections_mean(topography, shape)
topography = np.flip(topography, axis=0)


# Make training data
size = 10 # number of training samples generated
dt_train = PlanetDataSet(priors=priors, size=size, survey_framework=survey_framework, model_framework=model_framework, topography=topography)

# slim_output: reduced file size
# repeats: number of times a single matérn covariance function is repeated in the set
dt_train = dt_train.make_dataset(slim_output=True, repeats=5)

file_name = os.path.join(saveloc, f"trainset.pkl")
start_save = datetime.now()
with open(file_name, 'wb') as file:
    pkl.dump(dt_train, file)
print(f"Data set of size {size} made and saved as {file_name}.")



# Make validation data
size = 10 # number of training samples generated. Need to increase this for real use
dt_train = PlanetDataSet(priors=priors, size=size, survey_framework=survey_framework, model_framework=model_framework, topography=topography)

dt_train = dt_train.make_dataset(slim_output=True, repeats=5)

file_name = os.path.join(saveloc, f"validationset.pkl")
start_save = datetime.now()
with open(file_name, 'wb') as file:
    pkl.dump(dt_train, file)
print(f"Data set of size {size} made and saved as {file_name}.")


# Make test data
size = 10 # number of training samples generated. Need to increase this for real use
dt_train = PlanetDataSet(priors=priors, size=size, survey_framework=survey_framework, model_framework=model_framework, topography=topography)

dt_train = dt_train.make_dataset(slim_output=True, repeats=5)

file_name = os.path.join(saveloc, f"ppset.pkl")
start_save = datetime.now()
with open(file_name, 'wb') as file:
    pkl.dump(dt_train, file)
print(f"Data set of size {size} made and saved as {file_name}.")


