import pickle as pkl
import numpy as np
import os
import numpy as np

def read_files(data_location, filenames, datasize, model_parameters_to_include=['e_c', 'k_c', 'v_c'], noise=None):
    train_data, train_conditional = ([],[])
    if isinstance(filenames, str):
        filenames = [filenames]
    for f in filenames:
        with open(os.path.join(data_location, f), 'rb') as file:
            dt = pkl.load(file)
            td = np.vstack([dt[key] for key in model_parameters_to_include]).T
            # removing sh degrees below 2
            degrees = dt['sh_degrees']
            idx = np.argwhere(degrees[:,0]<2)
            tc = np.array(dt['gravity'])
            for i in idx:
                tc = np.delete(tc, i, 1)
            num_deg_ord = np.shape(tc)[1]
            tc = tc.reshape(tc.shape[0], -1)
            train_data.append(td)
            train_conditional.append(tc)
        print(f"{f} read")
    train_data = np.vstack(train_data)
    train_conditional = np.vstack(train_conditional)

    if noise is not None:
        noise = noise[:num_deg_ord,2:]
        noise_simulation = np.random.normal(loc=0.0, scale=noise, size=(np.shape(train_conditional)[0],)+np.shape(noise))
        noise_simulation = noise_simulation.reshape(noise_simulation.shape[0], -1)
        train_conditional = train_conditional + noise_simulation
        print("Added noise to the conditional...")
    return [train_data], [train_conditional]

