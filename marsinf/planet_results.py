import numpy as np
import matplotlib.pyplot as plt
from .utils import power_spectrum
from .dataset import PlanetDataSet
from .results import FlowResults
import os

plt.style.use('default')

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

    def power_spectra_comparison(self, original_range=None, filename='power_spectrum_comparison'):
        parameters_dict = self.priors.sample(size=np.shape(self.samples)[0], returntype='dict')
        for i, key in enumerate(self.parameter_labels):
            parameters_dict[key] = self.samples[:,i]

        results_dataset = PlanetDataSet(priors=self.priors, size=np.shape(self.samples)[0], model_framework=self.model_framework, survey_framework=self.survey_framework)

        results_dict = results_dataset.make_dataset(parameters_dict=parameters_dict, slim_output=True, repeats=1)
        sh_degrees = results_dict['sh_degrees']
        powerspectra = []
        for g in results_dict['gravity']:
            coeffs = np.c_[sh_degrees, g]
            ps, sh = power_spectrum(coeffs)
            idx_min = np.array(np.argwhere(sh<2))
            ps = np.delete(ps, idx_min, 0)
            sh = np.delete(sh, idx_min, 0)
            powerspectra.append(ps)
        powerspectra = np.vstack(powerspectra)
        min_spectrum = np.min(powerspectra, axis=0)
        max_spectrum = np.max(powerspectra, axis=0)
        mean_spectrum = np.mean(powerspectra, axis=0)
        std_spectrum = np.std(powerspectra, axis=0)
        plt.grid(zorder=0, linestyle='--')
        plt.fill_between(sh, original_range['min'][1:], original_range['max'][1:], label='Full Range', color='sandybrown', alpha=0.5, zorder=1)
        plt.fill_between(sh, mean_spectrum+std_spectrum/2, mean_spectrum-std_spectrum/2, label='SD', color='indianred', alpha=0.6, zorder=2)
        for i in range(10):
            if i == 0:
                plt.plot(sh, powerspectra[i,:], color='indianred', linestyle='--', zorder=3, label='Samples', linewidth=1)
            else:
                plt.plot(sh, powerspectra[i,:], color='indianred', linestyle='--', zorder=3, linewidth=1)
        plt.plot(sh, mean_spectrum, label='Mean', zorder=4, color='mediumpurple', linewidth=2)
        plt.legend()
        plt.xlim(left=0.0)
        plt.yscale('log')
        plt.ylabel('Power')
        plt.xlabel('SH degree')
        plt.savefig(os.path.join(self.directory, filename))
        plt.close()
