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
The only script you need to open is the [```main.py```](main.py) file. This script contains pre-written functions to calculate trajectories as desired, and to create porkchop plots.   
