#!/scratch/balta0/2263373r/conda_envs/giflow/bin/python
import torch
import numpy as np
import os
import pickle as pkl
import sys
import json

from marsinf.planet_results import PlanetFlowResults
from marsinf.flowmodel import FlowModel
from marsinf.utils import degree_variance
from marsinf.plot import plot_js_hist, plot_js_hist_and_scatter
from marsinf.datareader import DataReader
from marsinf.gravity import GravityMap
from marsinf.planet import Planet

model_parameters_to_include = ['e_c', 'k_c', 'v_c', 'e_m', 'k_m', 'v_m']
conditional_format = 'degree_variance'
plot_labels = [r'$\epsilon_c$', r'$\kappa_c$', r'$\sigma_c$', r'$\epsilon_m$', r'$\kappa_m$', r'$\sigma_m$']
min_degree = 7
flow_location = 'results'
data_location = 'data'


# -------------------- Reading the flow --------------------------
device = torch.device('cuda')
flow=FlowModel()
flow.load(flow_location)
flow.flowmodel.to(device)
flow.save_location = flow_location

# ------------------- Setup Information ----------------
with open(os.path.join(data_location, 'survey_framework.json'), 'r') as file:
    survey_framework = json.load(file)
with open(os.path.join(data_location, 'model_framework.json'), 'r') as file:
    model_framework = json.load(file)
with open(os.path.join(data_location, 'priors.pkl'), 'rb') as file:
    priors = pkl.load(file)
with open('noise_level_sh.pkl', 'rb') as file:
    noise = pkl.load(file)

# ------------------- PP test --------------------------

dr = DataReader(file_names='ppset.pkl', data_location=data_location, model_parameters_to_include=model_parameters_to_include, conditional_format=conditional_format, noise=noise, min_degree=min_degree)

pp_data, pp_conditional = dr.read_files()
ppsize = np.shape(pp_data[0])[0]

pp_dataset = flow.make_tensor_dataset(pp_data, pp_conditional, device=device, scale=True)

flow.pp_test(validation_dataset=pp_dataset,
        parameter_labels = plot_labels)

# -------------------- Results with real data ----------------------
# Loading the gravity data
with open('mars_gravity.pkl', 'rb') as file:
    data = pkl.load(file)

coeffs = np.c_[data['m'], data['n'], data['Smn'], data['Cmn']]
coeffs = np.concatenate((np.zeros((1,4)), coeffs), axis=0)

resolution = [4, 4]

long = np.arange(-180.0, 180.0, resolution[1])
lat = np.arange(-90.0, 90.0, resolution[0])
Long, Lat = np.meshgrid(long, lat)

shape = [np.shape(lat)[0], np.shape(long)[0]]

# trimming the coeffs arrays to agree with the latitude resolution
sh_max = shape[0]-1
coeffs = coeffs[:np.min(np.argwhere(coeffs[:,0]>sh_max)), :]
coeffs = coeffs[np.max(np.argwhere(coeffs[:,0]<min_degree))+1:, :]

gravity = GravityMap(lat=Lat, long=Long, resolution=resolution, coeffs=coeffs)
dv, _ = gravity.power_spectrum()

# Formatting it to agree with expected input
test_data = [np.zeros((1,1)) for i in range(len(model_parameters_to_include))] # random data, not used
test_conditional= [dv[np.newaxis,...]]

test_dataset = flow.make_tensor_dataset(test_data, test_conditional, device=device, scale=True)

# Sampling
num = 100000
samples, log_probabilities = flow.sample_and_logprob(test_dataset.tensors[1][0], num=int(num))
if conditional_format == 'coeffs_combined':
    half = int(np.shape(test_conditional[0][0])[0]/2)
    conditional = [test_conditional[0][0][:half], test_conditional[0][0][half:]]
else:
    conditional = [test_conditional[j][0] for j in range(len(test_conditional))]
result = PlanetFlowResults(samples=samples, conditional=conditional, log_probabilities=log_probabilities, parameter_labels=model_parameters_to_include, model_framework=model_framework, priors=priors, survey_framework=survey_framework)
result.directory = os.path.join(flow_location, f"real_data/")
with open(os.path.join(result.directory, f"results_{num}.pkl"), 'wb') as file:
    pkl.dump(result, file)

#  Corner plot                          
result.corner_plot(filename="corner_plot.png", labels=plot_labels)

# Volume reduction
samples, log_probabilities = flow.sample_and_logprob(test_dataset.tensors[1][0], num=int(1e7))
if conditional_format == 'coeffs_combined':
    half = int(np.shape(test_conditional[0][0])[0]/2)
    conditional = [test_conditional[0][0][:half], test_conditional[0][0][half:]]
else:
    conditional = [test_conditional[j][0] for j in range(len(test_conditional))]
result = PlanetFlowResults(samples=samples, conditional=conditional, log_probabilities=log_probabilities, parameter_labels=model_parameters_to_include, model_framework=model_framework, priors=priors, survey_framework=survey_framework)

prior_samples_scaled = flow.scalers['data'].scale_data(prior_samples_list, fit=False)
prior_logprobs = flow.logprob(torch.from_numpy(prior_samples_scaled.astype(np.float32)).to(device), test_dataset.tensors[1][0])
vr = result.get_volume_reduction(prior_logprobs)
print('Volume reduction:', vr)
