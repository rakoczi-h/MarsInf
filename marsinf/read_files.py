import pickle as pkl
import numpy as np
import os

def read_files(data_location, filenames, model_parameters_to_include=['e_c', 'k_c', 'v_c'], noise=None):
    """
    Function that read the files defined by data_location and filenames. This is specific for data files describing planets, which contain a dictionary with the parameters and the gravity of planets. Note that SH degrees below 2 will automatically be ignored.
    Parameters
    ----------
        data_location: str
            The folder where the data files are located.
        filenames: list
            Each element is a str describing the name of the file to be read.
        model_parameters_to_include: list
            The input files are searched for these keys and the parameters with these names are included in the data. (Default: ['e_c', 'k_c', 'v_c'])
        noise: np.ndarray
            Array containing the standard deviation of the noise that is added to the sh coefficients from the files. It is assoumed that the first two columns are the degrees and the order of each coefficient, the 3rd column is the mn and the 4th is the Cmn coefficients. Has to have same units as the data. If None, no noise is added. (Default: None)
    Output
        train_data: list
            Has one element, which is an ndarray with shape [dataset size, length of model_parameters_to_include].
        train_conditional: list
            Has one element, which is an ndarray with shape [dataset size, number of sh coefficients]
    """
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

