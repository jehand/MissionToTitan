# AEON
This repository contains pykep classes in order to optimize a trajectory to a celestial body and its moon (as long as the ephemeris data exists!).

<img align="left" width="46%" src="results/EVEES.gif">
<img align="right" width="50%" src="results/de1220_300.gif">
<br clear="both"/>

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
To run a DoE, the only script you need to open is the [```run_doe_mp_islands.py```](run_doe_mp_islands.py) file. This script contains pre-written functions to calculate trajectories as desired, and to run a DoE using Python's multiprocessing library. It does so using pygmo's archipelago, running each island on one core. This script can also be run on the cluster using the [```nonMpiDoE.pbs```](nonMpiDoE.pbs) file. At the bottom of the python file, one can specify the input filename with a list of sequences (for example, a one column .csv with a header row; each row written as 1123 where the numbers represent one planet to flyby; a dictionary input tells the program which number corresponds to which planet; these numbers just represent the flybys and the user can add a starting planet and ending planet as desired). The user also specifies the name of the output .csv, te departure window, the target moon, and the target orbit around this moon. 

Algorithms used in the archipelago can be changed in the [```define_algorithms.py```](define_algorithms.py) file under the ```interplanetary_algorithm``` and ```planetary_algorithm``` functions. The topology of the archipelago is also specified in the ```interplanetary_algorithm``` function, and can be changed if so desired. Make sure to download the correct spice kernels with the ephemeris data desired, and to update these filenames in the ```spice_kernels()``` and ```load_spice()``` functions in [```trajectory_solver.py```](trajectory_solver.py). 

Trajectory Solver also offers a ```pretty``` function that when provided the decision chromosome for the interplanetary phase and/or the planetary phase will print the results in human readable format. Additionally, there also exists a ```plot``` function that when provided both chromosomes will plot the trajectory and allow the user to view both phases at once. 

The [```algorithm_racing.py```](algorithm_racing.py) file allows one to race algorithms against each other on a user defined problem. The outputs of these algorithms are saved to a .csv that can then be analysed later. The DV and time of flight are output after the global algorithm, and overall, and the champion is returned if one wants to plot this after or print pretty results. When comparing archipelago algorithms, use the [```algorithm_racing_archipelago.py```](algorithm_racing_archipelago.py) file instead. Dictionaries at the top of both files define all the algorithms that pygmo currently offers, and the user can write the keys of this dictionary to a list at the bottom in order to test the specified algorithms. 

The [```gif_plotter.py```](gif_plotter.py) function allows one to plot gifs of their trajectory given a decision chromosome. The result can also be saved to show later. Furthermore, another animation function is provided in order to plot the trajectories given by an algorithm each generation. Both of these types of gifs can be seen above!

Lastly, the [```udp```](udps/) folder contains all the pykep defined problems (udp's). A user can add problems to this folder and then import them in Trajectory Solver to solve problems of their own needs. Note that the outputs might not be in the same order depending on the pykep base problem used, so Trajectory Solver might need to be edited, or a ```_compute_dvs``` function can be added to the udp to output results in the same order as the other udp's. Currently, there is a MGA, MGA_1DSM, bi-elliptic transfer (planetary system), a planetary system with 2 revolutions, and an electrical propulsion udp (not finished). The user can also define specific launch vehicles according to their v infinity at given declination angles and the payload they can carry to that v infinity in the [```rockets.py```](udps/rockets.py) file. This can then be imported intoa udp and used by the fitness function to output mass data for that trajectory. 

## Known Issues
When using SPICE kernels with pygmo's archipelago, initially it will print out saying that it does not have sufficient ephemeris data, however this will not affect the trajectory result and is just an artifact of the delayed loading of the kernels. 

## Examples
Example notebooks and past files are saved in the [```examples_tests```](examples_tests/) folder. These can be explored to see initial progress and notebook exploration of pykep's different examples. [```testing.ipynb```](examples_tests/testing.ipynb) is a good place to start if just starting off with pykep and pygmo.
