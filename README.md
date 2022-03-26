# MissionToTitan
This repository contains pykep classes in order to optimize a trajectory to Titan.

## Installation of Packages
You will need to have a working version of pykep in order to run the scripts in this repository. Pykep is available via [pip](https://pip.pypa.io/en/stable/):
```bash
pip install pykep
```
If this does not work, follow the instructions available [here](https://esa.github.io/pykep/installation.html) to install pykep from conda or from source. You will also need to install [pygmo](https://esa.github.io/pygmo2/install.html), which can also be done via [pip](https://pip.pypa.io/en/stable/).
```bash
pip install pygmo
```
Lastly, you should not need ```pygmo-plugins-nonfree``` to run the scripts in this repository. However, if desired, this library can be installed following the instructions [here](https://anaconda.org/conda-forge/pygmo_plugins_nonfree). 

## Usage
To run a DoE, the only script you need to open is the [```run_doe.py```](run_doe.py) file. This script contains pre-written functions to calculate trajectories as desired, and to run a DoE using Python's multiprocessing library. Therefore, this script can be run on the cluster. Make sure to download the correct spice kernels with the ephemeris data desired, and to update these filenames in the ```spice_kernels()``` and ```load_spice()``` functions in [```trajectory_solver.py```](trajectory_solver.py).
