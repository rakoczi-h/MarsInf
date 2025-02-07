import numpy as np
from scipy.spatial.distance import jensenshannon
import scipy.stats
import corner
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import matplotlib.gridspec as gridspec
import torch

from .prior import Prior
from .utils import get_colors, power_spectrum
from .dataset import PlanetDataSet

class FlowResults:
    """
    Class to incorporate methods of testing and visualising the results from a NF inversion.
    All parameters are assumed to have been rescaled to their original values. 
    Parameters
    ----------
        samples: np.ndarray
            An array of the samples from the flow, these contain parameter values. [num of samples, num of parameters]
        conditional: np.ndarray or torch.Tensor
            The conditional based on which these samples were generated. Essentially the gravity survey we're inverting
            If it is a torch tensor, it is converted to an np.ndarray
        survey_coordinates: np.ndarray
            The coordnates associated with the gravity data. [num of survey points, 3] (x, y, z is the order)
        log_probabilities: np.ndarray
            The log probability associated with each sample
        parameter_labels: list
            list of string containing the names of the inferred parameters
        true_parameters: np.ndarray
            The true values of the parameters, if known
        directory: str
            The location where the plots of the results are generated
    """
    def __init__(self, samples : np.ndarray, conditional, log_probabilities=None, parameter_labels=None, true_parameters=None, directory=None):
        self.nparameters = np.shape(samples)[1]
        self.nsamples = np.shape(samples)[0]
        if samples.ndim != 2:
            raise ValueError ('samples has to be 2D.')
        self.samples = samples
        self.conditional = conditional
        self.survey_coordinates = survey_coordinates
        self.parameter_labels = parameter_labels
        self.true_parameters = true_parameters
        self.log_probabilities = log_probabilities
        self.directory = directory

    def __setattr__(self, name, value):
        if name == 'log_probabilities':
            if value is not None:
                if self.nsamples != np.shape(value)[0]:
                    raise ValueError('The same number of samples and log_probabilities are required.')
                elif value.ndim != 1:
                    raise ValueError('log_probabilities has to be 1D.')
        if name == 'parameter_labels':
            if value is not None:
                if len(value) != self.nparameters:
                    print('Same number of labels are required are nparameters. Ignoring labels.')
                    value = None
        if name == 'directory':
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError("Expected str for directory name")
                if not os.path.exists(value):
                    os.mkdir(value)
        if name == 'conditional':
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
        super().__setattr__(name, value)

    def get_statistics_samples(self):
        stats = dict.fromkeys(['mode', 'q16', 'q50', 'q84'])
        idx = np.argwhere(self.log_probabilities == np.max(self.log_probabilities))
        idx = idx[0]
        stats['mode'] = self.samples[idx, :][0]
        sorted_idx = np.argsort(self.log_probabilities)
        q16 = int(self.nsamples*0.16)
        stats['q16'] = self.samples[q16,:]
        q50 = int(self.nsamples*0.5)
        stats['q50'] = self.samples[q50,:]
        q84 = int(self.nsamples*0.84)
        stats['q84'] = self.samples[q84,:]
        return stats

    def get_js_divergence(self, samples_to_compare, n=500, keep_wrong=True):
        """Function calculating the Jensen-Shannon divergence between the distribution of the samples of this class and another set of samples.
        The p(x) and q(x) functions are calculated using a KDE of the input samples.
        This is done for each dimension seperately.
        Parameters
        ----------
            samples_to_compare: array
                Samples from the other sampler. [no. of samples, no. of dimensions]. Assumed to be in the original data space (not normalised)
            n: int
                The number of gridpoints to consider when computing the kdes
        Output
        ------
            js: array of floats
                The list of JS-divergence values with length of the no. of parameters/dimensions.
        """
        if self.samples is None:
            raise AttributeError("Samples were not provided.")
        if not self.nparameters == np.shape(samples_to_compare)[1]:
            raise ValueError('The two sample sets do not have the same number of parameters.')
        js = []
        for i, dim in enumerate(self.samples.T):
            xmin = min([np.min(dim), np.min(samples_to_compare[:self.nsamples,i])])
            xmax = max([np.max(dim), np.max(samples_to_compare[:self.nsamples,i])])
            # calculate the minimum and maximum from both
            x_grid = np.arange(xmin, xmax+((xmax-xmin)/n), (xmax-xmin)/n) # the grid values we are using
            p = scipy.stats.gaussian_kde(dim)
            p_x = p.evaluate(x_grid)
            q = scipy.stats.gaussian_kde(samples_to_compare[:,i])
            q_x = q.evaluate(x_grid)
            js_pq = np.nan_to_num(np.power(jensenshannon(p_x, q_x), 2)) # jensenshannon gives the distance, need to square to get the divergence
            if js_pq > np.log(2):
                if keep_wrong:
                    js_pq = np.log(2)
                else:
                    print(f"Found wrong js values: {js_pq}")
                    return None
            js.append(js_pq)
        js = np.array(js)
        return js

    def get_volume_reduction(self, prior_logprobs, bounds=[0.16, 0.5, 0.84]):
        if self.samples is None:
            raise AttributeError("Samples were not provided as a class attribute.")
        if self.log_probabilities is None:
            raise AttributeError("Logprobabilities were not provided as a class attribute.")
        # Sort samples by logprob and find the value at the 90%
        sorted_logs = np.sort(self.log_probabilities)
        volume_reductions = []
        for b in bounds:
            num_bound = int(self.nsamples*b) # 90%
            log_bound = sorted_logs[num_bound]
            prior_bound_idx = np.argwhere(prior_logprobs > log_bound)
            volume_reductions.append(len(prior_bound_idx)/np.shape(prior_logprobs)[0])
        return volume_reductions


    def corner_plot(self, filename='corner.png', prior_bounds=None, labels=None):
        """Makes a simple corner plot with a single set of posterior samples.
        Parameter
        ---------
        filename: str
            The name under which it is saved
        scaler: sklearn.preprocessing scaler object
            Only used if self.samples does not exist.
        priors_bounds: list
            The length of the list is the same as the dimensions, and each element in the list is [minimum, maximum] bounds.
        """
        if np.shape(self.samples)[1] > 10:
            raise valueError(f"The samples have too many dimensions to present on a corner plot. Number of dimensions: {np.shape(self.samples)[1]}")
        plot_range = []
        if prior_bounds is None:
            for dim in self.samples.T:
                plot_range.append([min(dim), max(dim)])
        else:
            plot_range = prior_bounds
        if labels is None:
            if self.parameter_labels is None:
                labels = [f"q{x}" for x in range(self.nparameters)]
            else:
                labels = self.parameter_labels
        CORNER_KWARGS = dict(smooth=0.9,
                            show_titles=True,
                            label_kwargs=dict(fontsize=16),
                            title_kwargs=dict(fontsize=16),
                            quantiles=[0.16, 0.5, 0.84],
                            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                            plot_density=False,
                            plot_datapoints=False,
                            fill_contours=True,
                            max_n_ticks=3,
                            range=plot_range,
                            labels=labels,
                            color='sandybrown')

        figure = corner.corner(self.samples, **CORNER_KWARGS)
        if self.true_parameters is not None:
            values = self.true_parameters
            corner.overplot_lines(figure, values, color="black")
            corner.overplot_points(figure, values[None], marker="s", color="black")
        if self.directory is not None:
            plt.savefig(os.path.join(self.directory, filename), transparent=False)
        else:
            plt.savefig(filename, transparent=False)
        plt.close()
        print("Made corner plot...")

    # fix overlaid corners method !!
    def overlaid_corner(self, other_samples, dataset_labels = None, labels = None, filename='corner_plot_compare.png',  prior_bounds=None):
        """
        Plots multiple corners on top of each other
        Parameters
        ----------
            samples_list: list of arrays
                Contains samples from different inference algorithms
            parameter_labels: list
                The labels of the parameters over which the posterior is defined
            dataset_labels: list
                The name of the different methods the samples come from
            values: list
                The values of the true parameters, if not None then it is plotted over the posterior
            saveloc: str
                Location where the image is saved
            filename: str
                The name under which it is saved
            priors_bounds: list
                The length of the list is the same as the dimensions, and each element in the list is [minimum, maximum] bounds.
        Output
        ------
        image file
        """
        if not isinstance(other_samples, list):
            other_samples = [other_samples]
        _, ndim = other_samples[0].shape

        n = len(other_samples)+1
        colors = get_colors(n)
        samples_list = other_samples+[self.samples]
        max_len = max([len(s) for s in samples_list])
        plot_range = []
        if prior_bounds is None:
            for dim in range(ndim):
                plot_range.append(
                    [
                        min([min(samples_list[i].T[dim]) for i in range(n)]),
                        max([max(samples_list[i].T[dim]) for i in range(n)]),
                    ]
                )
        else:
            plot_range = prior_bounds
        if labels is None:
            if self.parameter_labels is None:
                labels = [f"q{x}" for x in range(self.nparameters)]
            else:
                labels = self.parameter_labels

        CORNER_KWARGS = dict(
        smooth=0.9,
        show_titles=False,
        label_kwargs=dict(fontsize=20),
        title_kwargs=dict(fontsize=20),
        quantiles=[0.16, 0.5, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False,
        plot_datapoints=False,
        fill_contours=True,
        max_n_ticks=3,
        range=plot_range,
        labels=labels)

        fig = corner.corner(
            samples_list[0],
            color=colors[0],
            **CORNER_KWARGS,
            hist_kwargs={'density' : True}
        )

        for idx in range(1, n):
            fig = corner.corner(
                samples_list[idx],
                fig=fig,
                weights=np.ones(len(samples_list[idx]))*(max_len/len(samples_list[idx])),
                color=colors[idx],
                **CORNER_KWARGS,
                hist_kwargs={'density' : True}
            )
        if self.true_parameters is not None:
            values = self.true_parameters
            corner.overplot_lines(fig, values, color="black")
            corner.overplot_points(fig, values[None], marker="s", color="black")
        if dataset_labels is not None:
            plt.legend(
                handles=[
                    mlines.Line2D([], [], 
                    color=colors[i],
                    label=dataset_labels[i])
                    for i in range(n)
                ],
                fontsize=20, frameon=False,
                bbox_to_anchor=(1, ndim), loc="upper right"
            )
        if self.directory is not None:
            plt.savefig(os.path.join(self.directory, filename), transparent=False)
        else:
            plt.savefig(filename, transparent=False)
        plt.close()
        print("Made corner plot...")

    def samples_to_csv(self, filename='samples.csv'):
        """
        Saves the samples into a csv.
        """
        df = DataFrame(self.samples)
        df.to_csv(os.path.join(self.directory, filename))

