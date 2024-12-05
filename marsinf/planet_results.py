import numpy as np
import matplotlib.pyplot as plt
from .utils import power_spectrum
from .dataset import PlanetDataSet
from .results import FlowResults
import os
import matplotlib.patheffects as pe

plt.style.use('default')
path_effects = [pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()]

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
        print(np.shape(samples))
        for i, key in enumerate(self.parameter_labels):
            low = self.priors.distributions[key][1]
            high = self.priors.distributions[key][2]
            print(low, high)
            idx = np.argwhere(samples[:,i]<low)
            samples = np.delete(samples, idx, axis=0)
            idx = np.argwhere(samples[:,i]>high)
            samples = np.delete(samples, idx, axis=0)
        print(np.shape(samples))
        return samples

    def power_spectra_comparison(self, original_range=None, filename='power_spectrum_comparison', enforce_prior_bounds=False):


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
            ps, sh = power_spectrum(coeffs)
            idx_min = np.array(np.argwhere(sh<2))
            ps = np.delete(ps, idx_min, 0)
            sh = np.delete(sh, idx_min, 0)
            powerspectra.append(ps)

        true_spectrum, sh = power_spectrum(np.c_[sh_degrees[3:,:], np.expand_dims(self.conditional[0], axis=1), np.expand_dims(self.conditional[1], axis=1)])
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
            plt.fill_between(sh, original_range['min'][1:], original_range['max'][1:], label='Full Range', color='gray', alpha=0.3, zorder=1)
        plt.fill_between(sh, min_spectrum, max_spectrum, label='Sample Range', color='sandybrown', alpha=0.4, zorder=2)
        plt.fill_between(sh, mean_spectrum+std_spectrum/2, mean_spectrum-std_spectrum/2, label='Sample SD', color='mediumpurple', alpha=0.5, zorder=3)
#        for i in range(5):
#            if i == 0:
#                plt.plot(sh, powerspectra[i,:], color='black', linestyle='--', zorder=4, label='Samples', linewidth=0.75)
#            else:
#                plt.plot(sh, powerspectra[i,:], color='black', linestyle='--', zorder=4, linewidth=0.75)
        plt.plot(sh, mean_spectrum, label='Mean', zorder=6, color='black', linewidth=1.25, path_effects=path_effects)
        plt.plot(sh, true_spectrum, label='Truth', zorder=5, color='firebrick', linewidth=1.25, path_effects=path_effects)
        plt.legend()
        plt.xlim(left=2.0, right=44.0)
        min_lim = np.min([np.min(min_spectrum), np.min(true_spectrum)])
        max_lim = np.max([np.max(max_spectrum), np.max(true_spectrum)])
        plt.ylim(min_lim, max_lim)
        plt.yscale('log')
        plt.ylabel('Power')
        plt.xlabel('SH degree')
        plt.savefig(os.path.join(self.directory, filename))
        plt.close()

    def power_spectra_comparison_v2(self, filename='power_spectra_sample_statistics_comparison.png'):
        mean_sample = np.mean(self.samples, axis=0)
        median_sample = np.median(self.samples, axis=0)
        samples = np.vstack([mean_sample, median_sample])
        print(np.shape(samples))

        parameters_dict = self.priors.sample(size=np.shape(samples)[0], returntype='dict')
        for i, key in enumerate(self.parameter_labels):
            parameters_dict[key] = samples[:,i]

        results_dataset = PlanetDataSet(priors=self.priors, size=np.shape(samples)[0], model_framework=self.model_framework, survey_framework=self.survey_framework)

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

        mean_spectrum = powerspectra[0]
        median_spectrum = powerspectra[1]

        true_spectrum, sh = power_spectrum(np.c_[sh_degrees[3:,:], np.expand_dims(self.conditional[0], axis=1), np.expand_dims(self.conditional[1], axis=1)])
        idx_min = np.array(np.argwhere(sh<2))
        true_spectrum = np.delete(true_spectrum, idx_min, 0)
        sh = np.delete(sh, idx_min, 0)
        plt.grid(zorder=0, linestyle='--')
        plt.fill_between(sh, true_spectrum, mean_spectrum, color='grey', alpha=0.5, zorder=1)
        plt.plot(sh, mean_spectrum, label='Mean', zorder=3, color='mediumpurple', linewidth=1.25, path_effects=path_effects)
        plt.plot(sh, median_spectrum, label='Median', zorder=4, color='firebrick', linewidth=1.25, path_effects=path_effects)
        plt.plot(sh, true_spectrum, label='Truth', zorder=2, color='black', linewidth=1.25, path_effects=path_effects)
        plt.legend()
        plt.xlim(left=2.0, right=44.0)
        min_lim = np.min([np.min(mean_spectrum), np.min(true_spectrum)])
        max_lim = np.max([np.max(mean_spectrum), np.max(true_spectrum)])
        plt.ylim(min_lim, max_lim)
        plt.yscale('log')
        plt.ylabel('Power')
        plt.xlabel('SH degree')
        plt.savefig(os.path.join(self.directory, filename))
        plt.close()


#    def christmas_tree_plot(self, filename='christmas_tree_plot.png'):
#        parameters_dict = self.priors.sample(size=np.shape(self.samples)[0], returntype='dict')
#        for i, key in enumerate(self.parameter_labels):
#            parameters_dict[key] = self.samples[:,i]
#
#        results_dataset = PlanetDataSet(priors=self.priors, size=np.shape(self.samples)[0], model_framework=self.model_framework, survey_framework=self.survey_framework)
#
#        results_dict = results_dataset.make_dataset(parameters_dict=parameters_dict, slim_output=True, repeats=1)
#        sh_degrees = results_dict['sh_degrees']
#
