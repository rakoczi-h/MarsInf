import os
import pickle as pkl
import numpy as np
import random

class DataReader():
    def __init__(self, file_names, data_location, model_parameters_to_include, noise=None, chunk_size=None):
        self.file_names = file_names
        self.n_files = int(len(file_names))
        self.data_location = data_location
        self.chunk_size = chunk_size
        self.model_parameters_to_include = model_parameters_to_include
        self.noise = noise

    def split_filenames(self, chunk_size=None, randomise=False):
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if self.chunk_size is None:
            filenames = self.file_names.copy()
            if randomise:
                return random.shuffle(filenames)
            else:
                return filenames
        else:
            filenames = self.file_names.copy()
            if randomise:
                random.shuffle(filenames)
            else:
                filenames = self.file_names.copy()
            num_sections = int(self.n_files/self.chunk_size)
            print(f"Datasize = {self.chunk_size*num_sections}")
            filenames = [filenames[(n*self.chunk_size):((n+1)*self.chunk_size)] for n in range(num_sections)]
            print(filenames)
            return filenames

    def read_files(self, filenames=None, noise_augment=False, noise_augment_factor=2):
        """
        Function that read the files defined by data_location and filenames. This is specific for data files describing planets, which contain a dictionary with the parameters and the gravity of planets. Note that SH degrees below 2 will automatically be ignored.
        Parameters
        ----------
            data_location: str
                The folder where the data files are located.
            filenames: list
                Each element is a str describing the name of the file to be read.
            noise: np.ndarray
                Array containing the standard deviation of the noise that is added to the sh coefficients from the files. It is assoumed that the first two columns are the degrees and the order of each coefficient, the 3rd column is the mn and the 4th is the Cmn coefficients. Has to have same units as the data. If None, no noise is added. (Default: None)
        Output
            train_data: list
                Has one element, which is an ndarray with shape [dataset size, length of model_parameters_to_include].
            train_conditional: list
                Has one element, which is an ndarray with shape [dataset size, number of sh coefficients]
        """


        if filenames is None:
            filenames = self.file_names
        train_data, train_conditional = ([],[])
        if isinstance(filenames, str):
            filenames = [filenames]
        for f in filenames:
            with open(os.path.join(self.data_location, f), 'rb') as file:
                dt = pkl.load(file)
                td = [dt[key] for key in self.model_parameters_to_include]
                # removing sh degrees below 2
                degrees = dt['sh_degrees']
                idx_min = np.array(np.argwhere(degrees[:,0]<2))
                tc = np.array(dt['gravity'])
                tc = np.delete(tc, idx_min, 1)
                #idx_max = np.array(np.argwhere(degrees[:,0]>20))-np.shape(idx_min)[0]
                #tc = np.delete(tc, idx_max, 1)
                num_deg_ord = np.shape(tc)[1]
                tc = [tc[:,:,0], tc[:,:,1]]
                train_data.append(td)
                train_conditional.append(tc)


        tc_list = []
        for i in range(len(train_conditional[0])):
            tc = []
            for j in range(len(train_conditional)):
                tc.append(train_conditional[j][i])
            tc = np.vstack(tc)
            if len(np.shape(tc)) == 1:
                tc = np.expand_dims(tc, axis=1)
            tc_list.append(tc)
        tc = []
        train_conditional = tc_list

        td_list = []
        for i in range(len(train_data[0])):
            td = []
            for j in range(len(train_data)):
                td.append(train_data[j][i])

            td = np.hstack(td)
            if len(np.shape(td)) == 1:
                td = np.expand_dims(td, axis=1)
            td_list.append(td)
        td = []
        train_data = td_list

        if self.noise is not None:
            self.noise = self.noise[:num_deg_ord,2:]
            if noise_augment:
                train_conditional = [np.repeat(tc, noise_augment_factor, axis=0) for tc in train_conditional]
                train_data = [np.repeat(td, noise_augment_factor, axis=0) for td in train_data]
            noise_simulation = np.random.normal(loc=0.0, scale=self.noise, size=(np.shape(train_conditional[0])[0],)+np.shape(self.noise))
            train_conditional[0] = train_conditional[0] + noise_simulation[:,:,0]
            train_conditional[1] = train_conditional[1] + noise_simulation[:,:,1]
            print("Added noise to the conditional...")

        return train_data, train_conditional
