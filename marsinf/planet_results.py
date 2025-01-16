import numpy as np
import matplotlib.pyplot as plt
from .utils import degree_variance
from .dataset import PlanetDataSet
from .results import FlowResults
import os
import matplotlib.patheffects as pe

plt.style.use('default')
path_effects = [pe.Stroke(linewidth=2.0, foreground="black"), pe.Normal()]

class PlanetFlowResults(FlowResults):
    def __init__(self, samples : np.ndarray, conditional, log_probabilities=None, parameter_labels=None, true_parameters=None, directory=None, model_framework=None, priors=None, survey_framework=None):
        self.nparameters = np.shape(samples)[1]
        self.nsamples = np.shape(samples)[0]
        if samples.ndim != 2:
            raise ValueError ('samples has to be 2D.')
        self.samples = samples
        self.conditional = conditional
        self.parameter_labels = parameter_labels
        self.true_parameters = true_parameters
        self.log_probabilities = log_probabilities
        self.directory = directory
        self.model_framework=model_framework
        self.survey_framework= survey_framework
        self.priors = priors

    def enforce_priors(self):
        samples = self.samples.copy()
        for i, key in enumerate(self.parameter_labels):
            low = self.priors.distributions[key][1]
            high = self.priors.distributions[key][2]
            idx = np.argwhere(samples[:,i]<low)
            samples = np.delete(samples, idx, axis=0)
            idx = np.argwhere(samples[:,i]>high)
            samples = np.delete(samples, idx, axis=0)
        return samples

    def power_spectra_comparison(self, original_range=None, filename='power_spectrum_comparison', enforce_prior_bounds=False, conditional_format=None):
        if enforce_prior_bounds:
            samples = self.enforce_priors()
        else:
            samples = self.samples

        parameters_dict = self.priors.sample(size=np.shape(samples)[0], returntype='dict')
        for i, key in enumerate(self.parameter_labels):
            parameters_dict[key] = samples[:,i]

        results_dataset = PlanetDataSet(priors=self.priors, size=np.shape(samples)[0], model_framework=self.model_framework, survey_framework=self.survey_framework)

        results_dict = results_dataset.make_dataset(parameters_dict=parameters_dict, slim_output=True, repeats=1)
        sh_degrees = results_dict['sh_degrees']
        powerspectra = []
        for g in results_dict['gravity']:
            coeffs = np.c_[sh_degrees, g]
            ps, sh = degree_variance(coeffs)
            idx_min = np.array(np.argwhere(sh<2))
            ps = np.delete(ps.flatten(), idx_min, 0)
            sh = np.delete(sh.flatten(), idx_min, 0)
            powerspectra.append(ps)

        if conditional_format == 'degree_variance':
            true_spectrum = self.conditional[0]
        else:
            true_spectrum, sh = degree_variance(np.c_[sh_degrees[3:,:], np.expand_dims(self.conditional[0], axis=1), np.expand_dims(self.conditional[1], axis=1)])
        idx_min = np.array(np.argwhere(sh<2))
        true_spectrum = np.delete(true_spectrum, idx_min, 0)
        sh = np.delete(sh, idx_min, 0)

        powerspectra = np.vstack(powerspectra)
        min_spectrum = np.min(powerspectra, axis=0)
        max_spectrum = np.max(powerspectra, axis=0)
        mean_spectrum = np.mean(powerspectra, axis=0)
        std_spectrum = np.std(powerspectra, axis=0)
        plt.grid(zorder=0, linestyle='--')
        if original_range is not None:
            plt.fill_between(sh, original_range['min'][1:], original_range['max'][1:], label='Data Set Range', color='gray', alpha=0.3, zorder=1)
        plt.fill_between(sh, min_spectrum, max_spectrum, label='Sample Range', color='sandybrown', alpha=0.7, zorder=2)
        plt.fill_between(sh, mean_spectrum+std_spectrum/2, mean_spectrum-std_spectrum/2, label='Sample SD', color='mediumpurple', alpha=0.7, zorder=3)

        plt.plot(sh, mean_spectrum, zorder=6, color='cornflowerblue', linewidth=1.25)
        plt.plot(sh, true_spectrum, zorder=5, color='firebrick')
        plt.scatter(sh, mean_spectrum, label='Mean', zorder=6, color='cornflowerblue', path_effects=path_effects, s=18)
        plt.scatter(sh, true_spectrum, label='Truth', zorder=5, color='firebrick', path_effects=path_effects, s=18)
        plt.legend()
        plt.xlim(left=2.0, right=44.0)
        min_lim = np.min([np.min(min_spectrum), np.min(true_spectrum)])
        max_lim = np.max([np.max(max_spectrum), np.max(true_spectrum)])
        plt.ylim(min_lim, max_lim)
        plt.yscale('log')
        plt.ylabel('Degree variance')
        plt.xlabel('SH degree')
        plt.savefig(os.path.join(self.directory, filename))
        plt.close()

