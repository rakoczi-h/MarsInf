import numpy as np
from oct2py import Oct2Py
import matplotlib.pyplot as plt
import os
plt.style.use('seaborn-v0_8')

class GravityMap():
    """
    Class for storing SH gravity data, translating to spatial maps and plotting spectra and maps.
    Parameters
    ----------
        g: dict
            The dictionary continuing the graviational acceleration vectors. (Default: None)
            Keys are 'X', 'Y', 'Z'
        grad: dict
            The dictionary contaning the graviational gradient tensors. (Default: None)
            Keys are 'xx', 'yy', 'zz', 'xy', 'xz', 'yz'
        lat, long: np.ndarrays
            These are assumed to be created with meshgrid, hence these are 2D arrays with shape [no. of points in latitude grid, no. of points in longitude grid] and reversed, respectively. The shape parameter is inferred using lat. Units: degrees (Default: None)
        coeffs: np.ndarray
            The SH coefficients. Has shape [4, number of coefficients]. The first two columns are the degree and order indices. (Default: None)
        height: float
            The height at which the gravity map is defined. Units in m. (Default: None)
        resolution: list
            The resolution in [latitude, longitude] in degrees. (Default: None)
    """
    def __init__(self, g=None, grad=None, lat=None, long=None, coeffs=None, height=None, resolution=None):
        self.g = g
        self.grad = grad
        self.lat = lat
        self.long = long
        self.coeffs = coeffs
        self.height = height
        self.shape = np.shape(lat)
        self.resolution = resolution
        self.ps = None
        self.sh_degrees = None

    def __setattr__(self, name, value):
        if name == 'g' or name == 'grad':
            if value is not None:
                if not isinstance(value, dict):
                    raise ValueError(f"{name} has to be a dictionary")
        super().__setattr__(name,value)

    def map_from_sh(self, radius=3.396*1e6, mass=6.4171*1e23, height=None):
        """
        Function that translates the SH coefficients into spatial values of the acceleration and gradient.
        NOT RECOMMENDED!
        Parameters:
        ----------
            radius: float
                The reference radius. Unit is m. (Default: 3.396*1e6)
            mass: float
                The mass of the planet. Unit is kg. (Default: 6.4171*1e23)
            height: float
                The height at which the map is computed. Unit is m. (Default: None)
                If it is None, the value is taken from the class.
        Output
        ------
            g: dict
                The dictionary continuing the graviational acceleration vectors.
                Keys are 'X', 'Y', 'Z'
            grad: dict
                The dictionary contaning the graviational gradient tensors.
                Keys are 'xx', 'yy', 'zz', 'xy', 'xz', 'yz'
        """
        if height is not None:
            self.height = height
        if self.height is None:
            raise ValueError('Need to provide the height of the survey')
        input_model = {'GM': 6.6743*1e-11*mass,
                        'Re': radius}
        octave = Oct2Py()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        octave.addpath(os.path.join(dir_path, 'gsh_tools/'))
        latLim = [np.min(np.min(self.lat)), np.max(np.max(self.lat)), self.resolution[0]]
        lonLim = [np.min(np.min(self.long)), np.max(np.max(self.long)), self.resolution[1]]

        SHbounds = [2.0, self.shape[0]-1] # truncating at 2nd degree for normalisation
        data = octave.model_SH_synthesis(lonLim,latLim,self.height,SHbounds,self.coeffs,input_model,nout=1)
        self.g = {'X': data.vec.X, 'Y': data.vec.Y, 'Z': data.vec.Z}
        self.grad = {'zz': data.ten.Tzz, 'xx': data.ten.Txx, 'yy': data.ten.Tyy, 'xy': data.ten.Txy, 'xz': data.ten.Txz, 'yz': data.ten.Tyz}
        octave.exit()
        return self.g, self.grad

    def power_spectrum(self):
        """
        Computes the power spectrum from the SH coefficients.
        Output
        ------
            ps: np.ndarray
                The power in each degree. Has the length of the number of SH degrees.
            sh_degrees: np.ndarray
                The SH degrees corresponding to each power value. Same length as ps.
        """
        sqrsum = self.coeffs[:,2]**2+self.coeffs[:,3]**2
        degrees = np.arange(np.min(self.coeffs[:,0]), np.max(self.coeffs[:,1])+1, 1)
        self.sh_degrees = degrees
        ps = []
        for l in degrees:
            c = 1/(2*l+1)
            idx = np.argwhere(self.coeffs[:,0]==l)
            ps.append(c*np.sum(sqrsum[idx]))
        self.ps = np.array(ps)
        return self.ps, self.sh_degrees

    def plot_spectrum(self, filename='gravity_spectrum.png'):
        """
        Creates a plot of the power spectrum.
        Parameters
        ----------
            filename: str
                The location where the image is saved.
        """
        plt.grid(zorder=-1)
        plt.plot(self.sh_degrees, self.ps)
        plt.scatter(self.sh_degrees, self.ps, marker='o')
        plt.xlabel('SH degree (l)')
        plt.ylabel('Power spectrum')
        plt.yscale('log')
        plt.savefig(filename)
        plt.close()

    def plot_field(self, filename='gravity.png'):
        """
        Creates a plot of the gravity map.
        Parameters
        ----------
            filename: str
                The location where the image is saved.

        """
        plot_arrays = [self.g['X'], self.g['Y'], self.g['Z'], self.grad['zz']]
        titles = [r'$g_x$', r'$g_y$', r'$g_z$', r'$g_{zz}$']
        proj = 'mollweide'
        fig, axes = plt.subplots(2,2, subplot_kw={'projection': proj})
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            to_plot = plot_arrays[i]
            im = ax.pcolormesh(self.long*np.pi/180.0, self.lat*np.pi/180.0, to_plot, cmap='rainbow')
            ax.set_title(titles[i])
            axpos = ax.get_position()
            pos_x = axpos.x0 # + 0.25*axpos.width
            pos_y = axpos.y0 - axpos.height/2 + 0.1
            cax_width = axpos.width
            cax_height = 0.01
            #create new axes where the colorbar should go.
            #it should be next to the original axes and have the same height!
            pos_cax = fig.add_axes([pos_x,pos_y,cax_width,cax_height])
            plt.colorbar(im, cax=pos_cax, orientation='horizontal')
        plt.savefig(filename)
        plt.close()
