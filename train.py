#!/scratch/balta0/2263373r/conda_envs/giflow/bin/python
import os
import pickle as pkl
import torch
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.decomposition import PCA
import numpy as np
from datetime import datetime
import sys
import matplotlib.pyplot as plt

from marsinf.scaler import Scaler
from marsinf.read_files import read_files
from marsinf.flowmodel import FlowModel, save_flow
from marsinf.datareader import DataReader

# ------------- Directories ---------------------------------
data_location = 'data/'  # THIS needs to be edited to give the data location
save_dir = 'results/' # THIS needs to be edited to give the saving location
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

model_parameters_to_include = ['e_c', 'k_c', 'v_c', 'e_m', 'k_m', 'v_m']
conditional_format = 'degree_variance'

# loading noise
with open('noise_level_sh.pkl', 'rb') as file:
    noise = pkl.load(file)

# ------------- Defining scalers ---------------------------
dr = DataReader(file_names="trainset.pkl", data_location=data_location, model_parameters_to_include=model_parameters_to_include, conditional_format=conditional_format, noise=noise, min_degree=7)

train_data, train_conditional = dr.read_files()
datasize = dr.datasize

scalers = [MinMaxScaler(), MinMaxScaler(), MinMaxScaler(), MinMaxScaler(), MinMaxScaler(), MinMaxScaler()]
sc_data = Scaler(scalers=scalers)
sc_data.scale_data(train_data, fit=True)

scalers = [QuantileTransformer(n_quantiles=10000, output_distribution='uniform')]
sc_conditional=Scaler(scalers=scalers, flatten=False)
sc_conditional.scale_data(train_conditional, fit=True)

scalers = {'conditional': sc_conditional, 'data': sc_data}

# ------------- Reading the data ----------------------------
dr_train = DataReader(file_names="trainset.pkl", data_location=data_location, model_parameters_to_include=model_parameters_to_include, noise=noise, conditional_format=conditional_format, min_degree=7)

dr_validation = DataReader(file_names="validationset.pkl", data_location=data_location, model_parameters_to_include=model_parameters_to_include, noise=noise, conditional_format=conditional_format, min_degree=7)

# ------------- Defining the prior ---------------------
with open(os.path.join(data_location, "priors.pkl"), 'rb') as file:
    priors = pkl.load(file)
# --------------- Defining the flow ------------------------
start_time = datetime.now()
save_location = os.path.join(save_dir, 'run_'+str(start_time))
os.mkdir(save_location)

device = torch.device('cuda')
# THIS needs to be edited for the hyperparameters of the flow
hyperparameters={'n_inputs': 6,
                 'n_conditional_inputs': 38,
                 'n_transforms': 23,
                 'n_blocks_per_transform': 8,
                 'n_neurons': 20,
                 'batch_norm': True,
                 'batch_size': 10000,
                 'early_stopping': True,
                 'dropout_probability': 0.0,
                 'lr': 0.001,
                 'epochs': 50000
}

flow = FlowModel(hyperparameters=hyperparameters, scalers=scalers)
flow.save_location = save_location
flow.data_location = data_location
flow.datasize = datasize # This assumes that the scaling data and the true dataset are equal
save_flow(flow)
flow.construct()

# Defining the flow inputs and training scheduler/optimiser
flow.optimiser = torch.optim.Adam(flow.flowmodel.parameters(), lr=flow.hyperparameters['lr'])
flow.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(flow.optimiser, mode='min', factor=0.05, patience=90, cooldown=10, min_lr=1e-6, verbose=True)

# ----------------- Training the flow ----------------------
flow.train( validation_datareader=dr_validation, train_datareader=dr_train, device=device, prior=priors)


