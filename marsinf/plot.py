import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from itertools import product
from collections import namedtuple
import imageio
import os
import matplotlib.patheffects as pe
plt.style.use('seaborn-v0_8-deep')
matplotlib.rcParams['axes.titlesize'] = 10
path_effects = [pe.Stroke(linewidth=2.0, foreground="black"), pe.Normal()]
path_effects_2 = [pe.Stroke(linewidth=1.0, foreground="black"), pe.Normal()]

def make_pp_plot(posterior_samples_list, truths, labels=None, filename=None, confidence_interval=[0.68, 0.95, 0.997],
                 lines=None, legend_fontsize='x-small', title=True,
                 confidence_interval_alpha=0.1, fig=None, ax=None,
                 **kwargs):
    """Create a pp_plot from sets of posterior samples and their corresponding injection values.

    Parameters
    ----------
    posterior_samples_list : list
        list of posterior samples sets
    truths : list
        list of dictionaries containing the true (injected) values for each observation corresponding to `posteror_samples_list`.
    filename : str, optional
        Filename to save pp_plot in, by default None (the plot is returned) (Default: None)
    confidence_interval : list, optional
        List of shaded confidence intervals to plot, (Default: [0.68, 0.95, 0.997])
    lines : list, optional
        linestyles to use, (Default: None (a default bank of linestyles is used))
    legend_fontsize : str, optional
        legend font size descriptor, by default 'x-small'
    title : bool, optional
        Display a title with the number of observations and a combined p-value, by default True
    confidence_interval_alpha : float, optional
        Transparency of the plotted confidence interval band, by default 0.1
    fig : Figure, optional
        Existing figure to overplot the p-p plot on, by default None (a Figure is created)
    ax : Axes, optional
        Existing axes to overplot the p-p plot on, by default None (axes are created)

    Returns
    -------
    figure : Figure
        the created (or existing, if fig is not None) matplotlib Figure object
    p_values : list
        the p-value for each parameter
    """

    credible_levels = list()
    for result, truth in zip(posterior_samples_list, truths):
        credible_levels.append(get_all_credible_levels(result, truth)
        )
    credible_levels = pd.DataFrame(credible_levels)
    if lines is None:
        colors = ["C{}".format(i) for i in range(8)]
        linestyles = ["-", "--", ":", "-."]
        lines = ["{}{}".format(a, b) for a, b in product(linestyles, colors)]
    if len(lines) < len(credible_levels.keys()):
        raise ValueError("Larger number of parameters than unique linestyles")

    x_values = np.linspace(0, 1, 1001)

    N = len(credible_levels)
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(4,4))


    if isinstance(confidence_interval, float):
        confidence_interval = [confidence_interval]
    if isinstance(confidence_interval_alpha, float):
        confidence_interval_alpha = [confidence_interval_alpha] * len(confidence_interval)
    elif len(confidence_interval_alpha) != len(confidence_interval):
        raise ValueError(
            "confidence_interval_alpha must have the same length as confidence_interval")

    for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
        edge_of_bound = (1. - ci) / 2.
        lower = scipy.stats.binom.ppf(1 - edge_of_bound, N, x_values) / N
        upper = scipy.stats.binom.ppf(edge_of_bound, N, x_values) / N
        # The binomial point percent function doesn't always return 0 @ 0,
        # so set those bounds explicitly to be sure
        lower[0] = 0
        upper[0] = 0
        ax.fill_between(x_values, lower, upper, alpha=alpha, color='k')

    pvalues = []
    for ii, key in enumerate(credible_levels):
        pp = np.array([sum(credible_levels[key].values < xx) /
                       len(credible_levels) for xx in x_values])
        pvalue = scipy.stats.kstest(credible_levels[key], 'uniform').pvalue
        pvalues.append(pvalue)
        if labels == None:
            try:
                name = posterior_samples_list[0].priors[key].latex_label
            except AttributeError:
                name = key
        else:
            name = labels[ii]
        label = "{} ({:2.3f})".format(name, pvalue)
        ax.plot(x_values, pp, lines[ii], label=label, **kwargs)
    Pvals = namedtuple('pvals', ['combined_pvalue', 'pvalues', 'names'])
    pvals = Pvals(combined_pvalue=scipy.stats.combine_pvalues(pvalues)[1],
                  pvalues=pvalues,
                  names=list(credible_levels.keys()))

    if title:
        ax.set_title("N={}, p-value={:2.4f}".format(
            len(posterior_samples_list), pvals.combined_pvalue))
    ax.set_xlabel("C.I.")
    ax.set_ylabel("Fraction of events in C.I.")
    ax.legend(handlelength=2, labelspacing=0.25, fontsize=legend_fontsize, loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=500)
        plt.close()

    return fig, pvals.pvalues, pvals.combined_pvalue

def compute_credible_level(posterior_samples, truth):
    """Get the 1-d credible interval for a truth value given a set of posterior samples

    Parameters
    ----------
    posterior_samples : ndarray
        Set of posterior samples
    truth : float
        truth value to get the C.I. for

    Returns
    -------
    credible_level : float
        The C.I. value
    """
    credible_level = np.mean(np.array(posterior_samples) < truth)
    return credible_level

def get_all_credible_levels(posterior_samples, truths):
    """Get credible levels for all parameters of this event/observation, returned as a dictionary.

    Parameters
    ----------
    posterior_samples : pandas DataFrame
        A dataframe where each parameter's posterior samples has its own column.
    truths : dict
        A dictionary of the truth values for this event/observation, with the same key naming convention as `posterior_samples`.

    Returns
    -------
    dict
        The credible intervals for each parameter for this set of posterior samples.
    """

    credible_levels = {key: compute_credible_level(posterior_samples[key], truths[key]) for key in list(posterior_samples)}
    return credible_levels


def plot_js_hist(js_divs, keys, filename='js_hist.png'):
    """
    Plots a histogram of js divergences between two distributions.
    Parameters
    ----------
        js_divs: array
             An array containing the js divergence values [no. of parameters, no. of js divergence values]
        keys: list
             List of strings containing the prameter names
        filename: str
             The name under which the file is saved. (Default: 'js_hist.png')
    Outputs
    -------
        counts: list of array
            The list containing the array of counts in each bin for the seperate parameters.
        bins: array
            The array containing the edges of the bins
        median: float
            The median of the overall distribution
    """
    if filename[-4:] != '.png':
        raise ValueError('The filetype for filename has to be .png')

    js_divs_list = []
    for i in range(np.shape(js_divs)[1]):
        js_divs_list.append(js_divs[:,i])
    js_divs_all = np.hstack(js_divs_list)
    median = np.median(js_divs_all)
    #counts, bins, _ = plt.hist(js_divs_list, bins=np.logspace(np.log10(0.0001), np.log10(0.6), 20), histtype='barstacked', range=(0, 0.6), density=False, label=keys)
    for i, jsl in enumerate(js_divs_list):
        counts, bins, _ = plt.hist(jsl, bins=10, histtype='step', density=False, label=keys[i])
    #plt.xscale('log')
    plt.legend()
    plt.xlabel("JS Divergence", fontsize=14)
    plt.ylabel("Counts", fontsize=14)
    plt.savefig(filename)
    plt.close()
    return counts, bins, median

def plot_js_hist_and_scatter(js_divs, parameters, keys, filename='js_hist_scatter.png'):
    if filename[-4:] != '.png':
        raise ValueError('The filetype for filename has to be .png')


    colors = ['sandybrown', 'indianred', 'cornflowerblue']
    fig, axes = plt.subplots(2,2, figsize=[10,10])
    ax = axes.flatten()
    js_divs_list = []
    for i in range(np.shape(js_divs)[1]):
        js_divs_list.append(js_divs[:,i])
    js_divs_all = np.hstack(js_divs_list)
    median = np.median(js_divs_all)

    #for i, jsl in enumerate(js_divs_list):
    ax[0].hist(js_divs_list, bins=20, histtype='barstacked', range=(0, 0.6), density=False, label=keys, color=colors)
    ax[0].hist(js_divs_all.flatten(), bins=20, histtype='step', range=(0, 0.6), density=False, color='black')
    ax[0].legend()
    ax[0].set_xlabel("JS Divergence", fontsize=14)
    ax[0].set_ylabel("Counts", fontsize=14)

    ax[1].grid(zorder=0, linestyle='--')
    ax[1].scatter(parameters[:,0], js_divs[:,0], zorder=2, color='black', s=25)
    ax[1].scatter(parameters[:,0], js_divs[:,0], zorder=2, color=colors[0], s=16)
    #ax[1].set_yscale('log')
    ax[1].set_ylabel("JS Divergence", fontsize=14)
    ax[1].set_xlabel(keys[0], fontsize=14)
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()

    ax[2].grid(zorder=0, linestyle='--')
    ax[2].scatter(parameters[:,1], js_divs[:,1], zorder=2, color='black', s=25)
    ax[2].scatter(parameters[:,1], js_divs[:,1], zorder=2, color=colors[1], s=16)
    #ax[2].set_yscale('log')
    ax[2].set_ylabel("JS Divergence", fontsize=14)
    ax[2].set_xlabel(keys[1], fontsize=14)

    ax[3].grid(zorder=0, linestyle='--')
    ax[3].scatter(parameters[:,2], js_divs[:,2], zorder=2, color='black', s=25)
    ax[3].scatter(parameters[:,2], js_divs[:,2], zorder=2, color=colors[2], s=16)
    #ax[3].set_yscale('log')
    ax[3].set_ylabel("JS Divergence", fontsize=14)
    ax[3].set_xlabel(keys[2], fontsize=14)
    ax[3].yaxis.set_label_position("right")
    ax[3].yaxis.tick_right()

    plt.savefig(filename)
    plt.close()

def make_gif(image_names, image_location='', filename='gif.gif'):
    """
    Makes a gif out of input images.
    Parameters
    ----------
        image_names: list
            The list of the names of the image files to read.
        image_location: str
            The directory where the images are located. (Default: '')
        filename: str
            The file under which the resulting gif is saved. (Default: 'gif.gif')
    """
    if filename[-4:] != '.gif':
        raise ValueError('The filetype for filename has to be .gif')

    images = []
    for image_name in image_names:
        images.append(imageio.imread(os.path.join(image_location, image_name)))
    imageio.mimsave(filename, images, fps=1)

