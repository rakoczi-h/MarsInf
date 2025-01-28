import torch
import numpy as np
import pandas as pd
from glasflow.flows import RealNVP, CouplingNSF
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib as mpl
from matplotlib import cm
import pandas as pd
import pickle as pkl
from glasflow.nflows.distributions import StandardNormal
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .latent import FlowLatent
from .plot import make_pp_plot
from .scaler import Scaler
from .prior import Prior
from .datareader import DataReader

plt.style.use('seaborn-v0_8-deep')

class FlowModel():
    """
    Class making a RealNVP flow model with methods to train and test it.
    Parameters
    ----------
        hyperparameters: dict
            Defines the parameters of the flow.
            keys:
                Relating to input:
                'n_inputs': The number of inputs per data point, so the number of parameters in the data model.
                'n_conditional_inputs': The number of conditional inputs per data poitn, so the number of survey points per survey.
                Relating to flow complexity:
                'n_transforms': Number of transforms in the flow.
                'n_block_per_transform': Number of blocks per transform.
                'n_neurons': Number of neurons per block.
                Relating to the training procedure:
                'epochs': The maximum number of epochs to train for.
                'batch_norm': The batches of data are individually normalised. (default: True)
                'batch_size': The number of data points given to to flow at each train iteration.
                'early_stopping': If the validation loss stops decreasing, training is automatically stopped. (default: False)
                'lr': The learning rate. (if there is a scheduler, this will only be the original lr)
        flowmodel: glasflow.flows.RealNVP model
        datasize: int
            Number of datapoints to use for training.
        scalers: dict
            keys: 'conditional', 'data'
                The scalers can be stored here to scale the data
        save_location: str
            The directory where the flow model and its outputs are saved.
    """
    def __init__(self, hyperparameters=None, flowmodel=None, datasize=None, scalers={"conditional": None, "data": None}, flowtype='RealNVP', optimiser=None, scheduler=None):
        self.flowmodel = flowmodel
        self.hyperparameters = hyperparameters
        self.datasize = datasize
        self.scalers = scalers
        self.loss = {"val": [], "train": []}
        self.save_location = ""
        self.data_location = ""
        self.flowtype = flowtype
        self.optimiser = scheduler
        self.scheduler = scheduler

    def __setattr__(self, name, value):
        if name == 'hyperparameters':
            if value is not None:
                if not isinstance(value, dict):
                    raise ValueError('Expected dict for hyperparameters.')
                value.setdefault('batch_norm', True)
                value.setdefault('early_stopping', False)
                value.setdefault('dropout_probability', 0.0)
        if name == 'save_location':
            if not value == '' and not os.path.exists(value):
                os.mkdir(value)
        if name == 'flowmodel':
            if value is not None:
                if not isinstance(value, RealNVP):
                    raise ValueError("flowmodel has to be a glasflow.flows.RealNVP object")
        super().__setattr__(name, value)

    def to_json(self):
        """
        Turns the flowmodel parameters into json file. The hyperparameters and the data size are incldued.
        """
        data = self.hyperparameters | {"datasize": self.datasize}
        with open(os.path.join(self.save_location, 'flow_info.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def getAttributes(self):
        """
        Helper function that returns only the useful attributes of the class.
        """
        return {name: attr for name, attr in self.__dict__.items()
                if not name.startswith("__") 
                and not callable(attr)
                and not type(attr) is staticmethod}

    def load(self, location: str, device=torch.device('cuda')):
        """
        Loads a saved and trained FlowModel.
        Parameters
        ----------
            location: str
                Directory where the FlowModel.pkl and the flow.pt files are located.
            device: torch.device (defaul: 'cuda')
                The device to send the network to.
        """
        with open(os.path.join(location, 'FlowModel.pkl'), 'rb') as file:
            attr = pkl.load(file)
        for key in list(attr.keys()):
            self.__setattr__(key, attr[key])
        self.construct()
        self.flowmodel.load_state_dict(torch.load(os.path.join(location, 'flow.pt')))
        self.flowmodel.to(device)

    def train(self, validation_datareader: DataReader, train_datareader: DataReader, scheduler=None, optimiser=None, device=torch.device('cuda'), prior=None):
        """
        The main training function, which trains, validates and plots diagnostics.
        Parameters
        ----------
            optimiser: torch.optim
                A torch optimiser object
            validation_dataset: torch.utils.data.TensorDataset
                The training dataset
            train_dataset: torch.utils.data.TensorDataset
                The validation dataset
            scheduler: torch.optim.lr_scheduler
                If provided, it is used to schedule the learning rate decay. Defaults to None.
            device: torch.device
                The device to send the flow to. Has to match that of the datasets. Defaults to cuda.
            prior: Prior object
                If given, it is used to calcualte JS divergence between prior and posterior as a metric.
        """
        # Setting up diagnostics folder:
        if not os.path.exists(os.path.join(self.save_location, 'diagnostics/')):
            os.mkdir(os.path.join(self.save_location, 'diagnostics/'))
        # Creating the flow
        if self.flowmodel is None:
            self.construct()

        if optimiser is not None:
            self.optimiser = optimiser
        if scheduler is not None:
            self.scheduler = scheduler
        if self.optimiser is None:
            raise ValueError('optimiser is required')

        print(f"Created flow and sent to {device}...")
        print(f"Network parameters:")
        print("----------------------------------------")
        print(f"n_inputs: \t\t {self.hyperparameters['n_inputs']}")
        print(f"n_conditional_inputs: \t {self.hyperparameters['n_conditional_inputs']}")
        print(f"n_transforms: \t\t {self.hyperparameters['n_transforms']}")
        print(f"n_blocks_per_trans: \t {self.hyperparameters['n_blocks_per_transform']}")
        print(f"n_neurons: \t\t {self.hyperparameters['n_neurons']}")
        print(f"batch_norm: \t\t {self.hyperparameters['batch_norm']}")
        print(f"dropout_probability: \t {self.hyperparameters['dropout_probability']}")
        print(f"batch_size: \t\t {self.hyperparameters['batch_size']}")
        print(f"optimiser: \t\t {type (self.optimiser).__name__}")
        print(f"scheduler: \t\t {type (self.scheduler).__name__}")
        print(f"early stopping: \t {self.hyperparameters['early_stopping']}")
        print(f"initial learning rate: \t {self.hyperparameters['lr']}")
        print("----------------------------------------")

        # if a chunksize is given for the dataloader, then the filenames are split into sublists
        if train_datareader.chunk_size is not None:
            train_filenames = train_datareader.split_filenames(randomise=True)
        else:
            train_data, train_conditional = train_datareader.read_files()
            train_dataset = self.make_tensor_dataset(train_data, train_conditional, device=device, scale=True)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.hyperparameters['batch_size'], shuffle=True)

        # Reading all the validation data in advance
        validation_data, validation_conditional = validation_datareader.read_files()
        validation_dataset = self.make_tensor_dataset(validation_data, validation_conditional, device=device, scale=True)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=self.hyperparameters['batch_size'], shuffle=True)

        self.flowmodel.to(device)
        loss_plot_freq = 10
        test_freq = 100
        # Training
        iters_no_improve = 0
        min_val_loss = np.inf
        start_train = datetime.now()
        for i in range(self.hyperparameters['epochs']):
            start_epoch = datetime.now()
            # Reading the training data in chunks
            if train_datareader.chunk_size is not None:
                train_loss = 0.0
                for tf in train_filenames:
                    train_data, train_conditional = train_datareader.read_files(filenames=tf)
                    train_dataset = self.make_tensor_dataset(train_data, train_conditional, device=device, scale=True)
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.hyperparameters['batch_size'], shuffle=True)
                    train_loss = self.train_iter(train_loader)
                    train_loss += train_loss
                self.loss['train'].append(train_loss/len(train_filenames))
            else:
                train_loss = self.train_iter(train_loader)
                self.loss['train'].append(train_loss)
            val_loss = self.validation_iter(validation_loader)
            self.loss['val'].append(val_loss)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self.loss['val'][-1])
                else:
                    self.scheduler.step()
            # Plotting the loss
            if not i % loss_plot_freq:
                self.plot_loss()
                data = {'train': self.loss['train'], 'val': self.loss['val']}
                df_loss = pd.DataFrame(data)
                df_loss.to_csv(os.path.join(self.save_location, 'loss.csv'))
            # Testing
            if not i % test_freq:
                torch.save(self.flowmodel.state_dict(), os.path.join(self.save_location, 'flow.pt'))
                print("----------------------------------------")
                start_test = datetime.now()
                print(f"Training time: \t {start_test-start_train}")
                print("Testing...")
                self.flowmodel.eval()
                latent_samples, latent_logprobs = self.forward_and_logprob(validation_dataset)
                latent_state = FlowLatent(latent_samples, log_probabilities=latent_logprobs)
                latent_state.get_kl_divergence_statistics()
                js_values = None
                if prior is not None:
                    js_values, js_mean = self.js_test(validation_dataset, prior=prior)
                self.plot_flow_diagnostics(latent_state, timestamp=start_test-start_train, js=js_values, filename=os.path.join('diagnostics', f"diagnostics_epoch_{i}.png"))
                end_test = datetime.now()
                print(f"Finished testing, time taken: \t {end_test-start_test}")
                print("----------------------------------------")
            # Setting early stopping condition
            if self.loss['val'][-1] < min_val_loss:
                min_val_loss = self.loss['val'][-1]
                iters_no_improve = 0
            else:
                iters_no_improve += 1
            if self.hyperparameters['early_stopping'] and iters_no_improve == 200:
                print("Early stopping!")
                break
            end_epoch = datetime.now()
            if not i % 10:
                print(f"Epoch {i} \t train: {self.loss['train'][-1]:.3f}   \t val: {self.loss['val'][-1]:.3f}   \t t: {end_epoch-start_epoch}")
        self.flowmodel.eval()
        print('Finished training...')
        end_train = datetime.now()

        torch.save(self.flowmodel.state_dict(), os.path.join(self.save_location, 'flow.pt'))

        self.plot_loss()
        latent_samples, latent_logprobs = self.forward_and_logprob(validation_dataset)
        latent_state = FlowLatent(latent_samples, log_probabilities=latent_logprobs)
        latent_state.get_kl_divergence_statistics()
        if prior is not None:
            js_values, js_mean = self.js_test(validation_dataset, prior=prior)
        self.plot_flow_diagnostics(latent_state, timestamp=start_test-start_train, js=js_values, filename='diagnostics_final.png')

        print(f"Run time: \t {end_train-start_train}")

    def train_iter(self, train_loader):
        """
        The training iteration function. A completion of one of these iteration is considered one epoch. Called in train function.
            optimiser: torch.optim
                The optimiser to use.
            validation_loader: torch.data.DataLoader
                The dataloader containing the validation data
            train_loader: torch.data.DataLoader
                The dataloader containing the training data
        """
        self.flowmodel.train()
        train_loss = 0.0
        for batch in train_loader:
            x, y = batch
            self.optimiser.zero_grad()
            _loss = -self.flowmodel.log_prob(x, conditional=y).mean()
            _loss.backward()
            self.optimiser.step()
            train_loss += _loss.item()
        train_loss = train_loss / len(train_loader)
        return train_loss

    def validation_iter(self, validation_loader):
        """
        The validation iteration function.C alled in train function after each epoch of training.
            validation_loader: torch.data.DataLoader
                The dataloader containing the validation data
        """
        self.flowmodel.eval()
        val_loss = 0.0
        for batch in validation_loader:
            x, y = batch
            with torch.no_grad():
                _loss = -self.flowmodel.log_prob(x, conditional=y).mean().item()
            val_loss += _loss
        val_loss = val_loss / len(validation_loader)
        return val_loss

    def construct(self):
        """
        Makes the RealNVP flow from the hyperparameters.
        """
        if self.flowtype == 'RealNVP':
            flow = RealNVP(
                n_inputs=self.hyperparameters['n_inputs'],
                n_transforms=self.hyperparameters['n_transforms'],
                n_conditional_inputs=self.hyperparameters['n_conditional_inputs'],
                n_neurons=self.hyperparameters['n_neurons'],
                n_blocks_per_transform=self.hyperparameters['n_blocks_per_transform'],
                batch_norm_between_transforms=self.hyperparameters['batch_norm'], #!
                dropout_probability = self.hyperparameters['dropout_probability']
            )
        elif self.flowtype == 'NSF':
            flow = CouplingNSF(
                n_inputs=self.hyperparameters['n_inputs'],
                n_transforms=self.hyperparameters['n_transforms'],
                n_conditional_inputs=self.hyperparameters['n_conditional_inputs'],
                n_neurons=self.hyperparameters['n_neurons'],
                n_blocks_per_transform=self.hyperparameters['n_blocks_per_transform'],
                batch_norm_between_transforms=self.hyperparameters['batch_norm'], #!
                dropout_probability = self.hyperparameters['dropout_probability'],
                distribution = self.hyperparameters['distribution'],
                num_bins = self.hyperparameters['num_bins']
            )
        else:
            raise ValueError('flowtype can be NSF or RealNVP')
        self.flowmodel = flow
        return flow

    def js_test(self, dataset: torch.utils.data.TensorDataset, prior: Prior):
        js_values = []
        js_mean = []
        for i in range(100):
            samples, _ = self.sample_and_logprob(dataset.tensors[1][i], num=2000)
            js, mean_js = prior.get_js_divergence(samples, n=100, num_samples=2000)
            js_values.append(js)
            js_mean.append(mean_js)
        js_mean = np.mean(js_mean)
        js_values = np.vstack(js_values)
        return js_values, js_mean


    # --------------------- Plotting Methods -------------------------------------
    def plot_loss(self):
        """
        Makes a plot of the training and validation loss wrt. number of epochs.
        """
        plt.plot(self.loss['train'], label='Train')
        plt.plot(self.loss['val'], label='Val.')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.save_location, "loss.png"))
        plt.close(
)
    def plot_flow_diagnostics(self, latent: FlowLatent, timestamp=None, js=None, filename='diagnostics.png'):
        """
        Plots diagnostics during training.
        Parameters:
            latent: FlowLatent
                Used to generate plots relating to the latent space
            timestamp:
                Any value we want to pass to be printed as an indication of timestamp. Defaults to None.
            js: np.ndarray
                with the shape [number of test cases, number of dimensions]. If given, this is used to make a js divergence metric histogram.
        """
        if js is not None:
            plt.figure(figsize=(20,50))
            fig, axs = plt.subplot_mosaic([['A', 'A'], ['B', 'B'], ['C', 'C'], ['D', 'E']],
                                  width_ratios=np.array([1,1]), height_ratios=np.array([1,1,1,1.5]),
                                  gridspec_kw={'wspace' : 0.3, 'hspace' : 1.0})
        else:
            plt.figure(figsize=(20,30))
            fig, axs = plt.subplot_mosaic([['A', 'A'], ['B', 'B'], ['D', 'E']],
                      width_ratios=np.array([1,1]), height_ratios=np.array([1,1,1]),
                      gridspec_kw={'wspace' : 0.1, 'hspace' : 0.3})
        # Plotting the loss
        ax = axs['A']
        ax.set_box_aspect(0.2)
        ax.plot(self.loss['train'])
        ax.plot(self.loss['val'])
        ax.set_xlabel('Epoch', fontdict={'fontsize': 10})
        ax.set_ylabel('Loss', fontdict={'fontsize': 10})
        if timestamp is not None:
            ax.set_title(f"Loss | {timestamp}", fontdict={'fontsize': 10})
        else:
            ax.set_title(f"Loss", fontfict={'fontsize': 10})

        # Plotting the latent space distribution
        ax = axs['B']
        for i in range(np.shape(latent.samples)[1]):
            if i == 0:
                ax.hist(latent.samples[:,i], bins=100, histtype='step', density=True)
            else:
                ax.hist(latent.samples[:,i], bins=100, histtype='step', density=True)
        g = np.linspace(-4, 4, 100)
        ax.plot(g, norm.pdf(g, loc=0.0, scale=1.0), color='navy')
        ax.set_box_aspect(0.2)
        ax.set_xlim(-4, 4)
        ax.set_title(f"Latent Space Distribution | Mean KL = {latent.kl_divergence['mean']:.3f}", fontdict={'fontsize': 10})
        ax.set_ylabel('Sample Density', fontdict={'fontsize': 10})

        # Plotting the latent space distribution
        if js is not None:
            ax = axs['C']
            for i in range(np.shape(js)[1]):
                ax.hist(js[:,i], bins=10, histtype='step', density=True)
            ax.set_box_aspect(0.2)
            ax.set_title(f"JS Divergence with Prior | Mean JS = {np.mean(js):.3f}", fontdict={'fontsize': 10})
            ax.set_ylabel('Count Density', fontdict={'fontsize': 10})

        # Plotting a histogram of the latent log probabilites
        ax = axs['D']
        ax.hist(latent.log_probabilities, bins=100, density=True)
        ax.set_box_aspect(1)
        ax.set_title('LS Sample Probabilities', fontdict={'fontsize': 10})
        ax.set_ylabel('Prob Density', fontdict={'fontsize': 10})
        ax.set_xlabel('Log-Prob', fontdict={'fontsize': 10})

        # Plotting an image of the correlation of the latent space samples
        ax = axs['E']
        ax.set_box_aspect(1)
        sigma = np.abs(np.corrcoef(latent.samples.T))
        im = ax.imshow(sigma, norm=mpl.colors.LogNorm())
        ax.set_title('LS Correlation', fontdict={'fontsize': 10})
        cbar_ax = fig.add_axes([0.8, 0.1, 0.02, 0.2]) # left, bottom, width, height
        fig.colorbar(im, cax=cbar_ax, label='corr coeff')

        plt.savefig(os.path.join(self.save_location, filename), transparent=False)
        plt.close()

    # ------------------------ Drawing samples ------------------------------------
    def forward_and_logprob(self, dataset: torch.utils.data.TensorDataset, num=None):
        """
        Drawing samples from the latent space and returning their corresponding log probabilties too
        Parameters
        ----------
            dataset: tensor dataset
                The data set containg the samples which we want to forward model to the latent space.
            num: int
                the number of samples to draw (Defaults to None)
        Output
        ------
            z_: array
                Laten space samples. Same shape as data.
            log_prob: array
                The log probabilties of each sample. [no. of samples]
        """
        self.flowmodel.eval()
        if num is None:
            num = dataset.tensors[0].shape[0]
        if num > dataset.tensors[0].shape[0]:
            raise ValueError('More samples requested than data samples provided')
        with torch.no_grad():
            z_, log_prob = self.flowmodel.forward_and_log_prob(dataset.tensors[0][:int(num)], conditional=dataset.tensors[1][:int(num)])
        z_ = z_.cpu().numpy()
        log_prob = log_prob.cpu().numpy()
        return z_, log_prob

    def sample_and_logprob(self, conditional: torch.Tensor, num=1):
        """
        Drawing samples from the posterior and return their corresponding log probabilites.
        Parameters
        ----------
            conditional: torch.Tensor
                The conditional based on which we want to sample. [num_conditionals, lenght of conditional]. Can pass multiple conditionals.
            num: int
                Number of samples to draw per conditional. (Default: 1)
        """
        self.flowmodel.eval()
        if conditional.dim() == 1:
            conditional = torch.unsqueeze(conditional, dim=0)
        conditional = torch.repeat_interleave(conditional, num, axis=0)
        with torch.no_grad():
            start_sample = datetime.now()
            s, l = self.flowmodel.sample_and_log_prob(num, conditional=conditional)
            end_sample = datetime.now()
        # print(f"{num} samples drawn. Time taken: \t {end_sample-start_sample}")
        s = s.cpu().numpy()

        s = self.scalers['data'].inv_scale_data(s)
        s = np.hstack(s)
        l = l.cpu().numpy()
        return s, l


    # --------------------------- Testing --------------------------------------
    def pp_test(self, validation_dataset: torch.utils.data.TensorDataset, num_samples=2000, num_cases=100, num_params=10, parameter_labels=None, filename='pp_plot.png'):
        """
        Draws samples from the flow and constructs a p-p plot.
        Parameters
        ----------
            validation_dataset: torch.utils.data.TensorDataset
                The data set for which we want to compute the pp values
            num_samples: int
                Number of samples to draw for each test case (Default: 2000)
            num_cases: int
                The number of test cases to consider (Default: 100)
            num_params: int
                The number of parameters to plot (Default: 10)
            parameter_labels: list of str
                The name of the parameters. If not given, then the names will be automatically generated to be ['q1', 'q2', ...] (Default: None)
            filename: str
                The location where the image is saved. (Default: 'pp_plot.png')
        """
        truths = validation_dataset.tensors[0][:int(num_cases)].cpu().numpy()
        truths = self.scalers['data'].inv_scale_data(truths)
        if len(truths) > num_params:
            indices = np.random.randint(len(truths), size=num_params)
        else:
            num_params = len(truths)
            indices = np.arange(0, num_params)
        if parameter_labels == None:
            parameter_labels = [f"q{x}" for x in range(num_params)] # number of parameters to get the posterior for (will be 512)
        posteriors = []
        injections = []
        with torch.no_grad():
            for cnt in range(num_cases):
                posterior = dict()
                injection = dict()
                x, _ = self.sample_and_logprob(conditional=validation_dataset.tensors[1][cnt], num=num_samples)
                for i, key in enumerate(parameter_labels):
                    posterior[key] = x[:,indices[i]]
                    injection[key] = truths[indices[i]][cnt,:].flatten()
                posterior = pd.DataFrame(posterior)
                posteriors.append(posterior)
                injections.append(injection)
        print("Calculated results for p-p...")
        _, pvals, combined_pvals = make_pp_plot(posteriors, injections, filename=os.path.join(self.save_location, filename), labels=parameter_labels)
        print("Made p-p plot...")
        return pvals, combined_pvals

    def saliency(self, conditionals, delta=0.1, num=1000, bandwidth=1, filename='saliency.png', parameter_labels=None, device=torch.device('cuda'), include_logprob=False, min_degree=2):
        """
        Performs a saliency test with a self-defined parameter importance metric.
        Parameters
        ----------
            conditionals: list or np.ndarray
                The conditional inputs for which the saliency test is perfomed. If it's a list, it is assumed that the different elements of the list are all independent conditional inputs in the right format so that they can be turned into tensors and used to sample from the network. If it is an array, and it is 1D, it is assumed that it is just one conditional. If it is an array, and it is 2D, the first dimension is assumed to represent the number of conditionals.
            delta: int, float or list
                The amount by which the elements of a single conditional input are altered when measuring the sensitivity of the output. If it is a list, then the test is performed for all values individually. (Default: 0.1)
            num: int
                The number of samples to draw for each conditional input. (Default: 1000)
            bandwidth: int
                The number of elements of the conditional to alter at once. If the length of the conditional is not divisible by this number, the rest of the conditional elements are ignored. (Default: 1)
            filename: str
                The name under which the plot is saved. (Default: 'saliency.png')
            parameter_labels: list
                The labels of the aprameters in the output. If not given, it is automatically generated. (Default: None)
        """
        # Making simulated parameter labels if none are given
        if parameter_labels == None:
            parameter_labels = [f"q{x}" for x in range(num_params)]
        if include_logprob:
            parameter_labels.append('Log probability')

        # Checking the type for the conditional.
        if isinstance(conditionals, np.ndarray):
            if conditionals.ndim == 1:
                    conditionals = conditionals[np.newaxis,...]     # if it is a 1D ndarray, it is assumed that that is one conditional.
            conditionals = [conditionals[i,...] for i in range(np.shape(conditionals)[0])] # if a multidimensional array, then it is assumed that the first axis signifies the different conditonals
        if not isinstance(conditionals, list):
            raise ValueError('conditionals has to be a list or np.ndarray')

        num_sections = int(np.shape(conditionals[0])[0]/bandwidth)

        if isinstance(delta, float) or isinstance(delta, int):
            delta = [delta]
        if not isinstance(delta, list):
            raise ValueError('delta has to be list, int or float')

        with torch.no_grad():
            saliency_deltas = []
            for dj, d in enumerate(delta):
                saliency_conditionals = []
                for j, c in enumerate(conditionals):
                    # defining the conditional
                    conditional = torch.from_numpy(c).to(device)
                    if conditional.dim() == 1:
                        conditional = torch.unsqueeze(conditional, dim=0)
                    conditional = torch.repeat_interleave(conditional, num, axis=0)
                    # defining the base distribution and sampling it
                    base_distribution = StandardNormal([self.hyperparameters['n_inputs']])
                    latent_samples = base_distribution.sample(num).to(device)
                    # getting the corresponding samples from the data space
                    q0, _ = self.flowmodel.inverse(latent_samples.to(dtype=torch.float), conditional=conditional.to(dtype=torch.float)) # this gives log|J|, not logprob
                    lq0 = self.flowmodel.log_prob(q0, conditional=conditional.to(dtype=torch.float)) # calling log prob too
                    q0 = q0.cpu().numpy()
                    lq0 = lq0.cpu().numpy()
                    if include_logprob:
                        q0 = np.c_[q0, lq0[...,np.newaxis]] # joining the samples and the log probabilities into one array

                    saliency = []
                    for i in range(num_sections):
                        conditional_new = np.zeros(conditional.shape[1])
                        # defining the new conditional
                        conditional_new[i*bandwidth:((i+1)*bandwidth)] = np.ones(bandwidth)*d
                        # the elements extending beyond the range of num_sections*bandwidth are ignored
                        conditional_new = torch.from_numpy(conditional_new).to(device)
                        conditional_new = torch.unsqueeze(conditional_new, dim=0)
                        conditional_new = torch.repeat_interleave(conditional_new, num, axis=0)
                        conditional_new = conditional_new + conditional
                        #if j == 0:
                        #    plt.plot(conditional_new[0,:].cpu().numpy())
                        # getting the corresponding samples, for the same latent locations as before
                        q, _ = self.flowmodel.inverse(latent_samples.to(dtype=torch.float), conditional=conditional_new.to(dtype=torch.float))
                        q = q.cpu().numpy()
                        if include_logprob:
                            lq = self.flowmodel.log_prob(torch.from_numpy(q0[:,:-1]).to(device), conditional=conditional_new.to(dtype=torch.float)) # calling the log prob on the original samples
                            lq = lq.cpu().numpy()
                            q = np.c_[q, lq[..., np.newaxis]]
                        # defining the saliency metric
                        sal = np.sqrt((q0-q)**2)/d
                        sal = np.mean(sal, axis=0)
                        saliency.append(sal)
                    saliency = np.array(saliency)
                    saliency = np.repeat(saliency, bandwidth, axis=0)
                    saliency_conditionals.append(saliency)
                saliency = np.array(saliency_conditionals)
                saliency = np.mean(saliency, axis=0) # taking the mean across all the conditionals
                #saliency = (saliency-np.min(saliency, axis=0))/(np.max(saliency, axis=0)-np.min(saliency, axis=0)) # normalising to the range of 0 to 1
                saliency[np.isnan(saliency)] = 0.0
                saliency_deltas.append(saliency)
        saliency_deltas_arr = np.array(saliency_deltas)
        saliency_deltas_arr = (saliency_deltas_arr-np.min(saliency_deltas_arr))/(np.max(saliency_deltas_arr)-np.min(saliency_deltas_arr))+1e-10
        num_subplots = np.shape(saliency_deltas_arr)[2]
        cmap = mpl.colormaps['plasma']
        #norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        print(np.min(saliency_deltas_arr))
        print(np.max(saliency_deltas_arr))
        norm = mpl.colors.LogNorm(vmin=1e-2, vmax=1.0)
        fig, ax = plt.subplots(num_subplots, 1, figsize=(10,num_subplots*4), gridspec_kw={'hspace': 0.1})
        for j in range(num_subplots):
            if not (j == num_subplots-1):
                ax[j].set_xticks(ticks=[])
            else:
                ax[j].set_xlabel('SH Degree', fontdict={'fontsize': 14})
                ax[j].set_xticks(np.arange(0, np.shape(saliency_deltas_arr)[1], 5), labels=np.arange(0, np.shape(saliency_deltas_arr)[1], 5)+min_degree)
            ax[j].imshow(saliency_deltas_arr[:,:,j], norm=norm, cmap=cmap)
            ax[j].set_yticks(np.arange(0, len(delta), 2), labels=[f"{delta[d]:.0e}" for d in np.arange(0, len(delta), 2)])
            ax[j].set_ylabel(r'$\Delta$d', fontdict={'fontsize': 14})
            ax[j].set_title(parameter_labels[j], fontdict={'fontsize': 14})
        axpos = ax[j].get_position()
        pos_x = axpos.x0
        pos_y = axpos.y0 - 0.07
        cax_width = axpos.width
        cax_height = 0.02
        cax = fig.add_axes([pos_x, pos_y, cax_width, cax_height])
        cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', label='Normalised importance')
        cb.set_label(label='Normalised importance', fontsize=14)
        plt.savefig(os.path.join(self.save_location, filename))
        plt.close()
        return saliency


    # --------------------------- Dataset ---------------------------------------
    def make_tensor_dataset(self, data, conditional, device=torch.device('cuda'), scale=True):
        """
        Makes tensor dataset.
        Parameters
        ----------
            data: np.array
                The array containing the data points. The output from the BoxDataSet.make_data_arrays() method is suitable.
            conditional: np.array
                The array containing the conditionals.
            device: torch.device
                Has to match with the one provided to train() (Default: 'cuda')
            scale: bool
                If true, the data given to the function will be scaled before it is turned into a datalaoder. (default: True)
        """
        if scale:
            if self.scalers['conditional'] is None:
                raise ValueError("The conditional scaler was not given")
            if self.scalers['data'] is None:
                raise ValueError("The data scaler was not given")
            data = self.scalers['data'].scale_data(data, fit=False)
            conditional = self.scalers['conditional'].scale_data(conditional, fit=False)
            #conditional_size = conditional.shape[0]
            #conditional = self.scalers['conditional'].transform(conditional.reshape(-1, conditional.shape[-1]))
            #conditional = conditional.reshape(conditional_size, -1)
        x_tensor = torch.from_numpy(data.astype(np.float32)).to(device)
        y_tensor = torch.from_numpy(conditional.astype(np.float32)).to(device)
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        return dataset

def save_flow(flow : FlowModel):
    """
    Function to save flowmodel as a pkl file and info about it in a json.
    """
    flow.to_json()
    with open(os.path.join(flow.save_location, 'FlowModel.pkl'), 'wb') as file:
        pkl.dump(flow.getAttributes(), file)
