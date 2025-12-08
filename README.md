# MarsInf

Software for creating simulated models of the topography and density distributions of the crust and mantle of Mars, and for computing the gravitational field of these toy models. Finally, a parameter inference tool applying Normalising Flows is implemented for the inference of the parameters of the planet.

## Installation

Installation guide for Linux.

```bash
git clone https://github.com/rakoczi-h/MarsInf.git
cd MarsInf 
```

Two separate conda environments are required for creating planet simulations and for inference due to package conflicts.

1. Environment for planet simulations, containing octave-python interpreter:
```bash
conda env create -f octave_environment.yml
conda activate octave
conda env config vars set OCTAVE=$CONDA_PREFIX/bin/octave-cli-8.4.0
conda env config vars set OCTAVE_EXECUTABLE=$CONDA_PREFIX/bin/octave-cli-8.4.0
conda deactivate
conda activate octave
```
Note: Requires GNU Octave installation.

2. Environment for inference, containing pytorch libraries:
```bash
conda env create -f flow_environment.yml
conda activate flow
```

## Usage
Can make training and validation data using scripts such as `make_planet_dataset.py`.

Example training script is included in `train.py`.

Examples of how to create the p-p plot for validation and how to sample and plot the posterior distribution is included in `test.py`.


## Acknowledgements
The author acknowledges the use of the [_nflows_](https://github.com/uofgravity/nflows#citing-nflows) (C. Durkan et al. (2019)) package to construct implement the normalising flow elements and the [_glasflow_](https://github.com/uofgravity/glasflow) (Williams et al. (2023)) package to construct the neural network. The Matlab code used for planetary modelling belongs to Bart Root [_GSH_](https://github.com/bartroot/GSH) and was slightly altered to be suitable for usage with Octave.

The data that are used in this study is derived from data available on PDS Geosciences Node, [_topography_](https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=MGS-M-MOLA-5-MEGDR-L3-V1.0) (G. Neumann et al. (2003)) and [_gravity_](https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=MRO-M-RSS-5) (M. Zuber et al. (2010)).
