"""
Porkchop Plotter
AEON Grand Challenge
Spring 2022
Sarah Hopkins
"""

# General Imports
import pykep as pk
import pygmo as pg
from datetime import datetime as dt
import numpy as np

# Plotting imports
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import os
from multiprocessing import Pool, cpu_count, set_start_method
from csv import DictWriter, DictReader
from udps.chemical_mga import TitanChemicalMGAUDP
from udps.planetary_system2 import PlanetToSatellite
from trajectory_solver import TrajectorySolver, load_spice, spice_kernels
from itertools import repeat
from datetime import datetime as dt
from display_style import bcolors
from ast import literal_eval
import pykep as pk
from pykep.planet import jpl_lp

'''
#######################################################################################################################
Load Spice Kernel and Relative Planets
#######################################################################################################################
'''

# Instantiate the class once from the start and don't do it again
interplanetary_udp = TitanChemicalMGAUDP
planetary_udp = PlanetToSatellite
trajectory = TrajectorySolver(interplanetary_udp, planetary_udp)

'''
#######################################################################################################################
Porkchop Plotter Function
#######################################################################################################################
'''
def porkchop_plotter(traj_solver, T0, planet_sequence, Vinf_dep, e_target, tof_bounds, n, t_step, out_filename):
    # traj_solver:      the trajectory solver function        
    # T0:               launch date
    # planet_sequence:  sequence of planets for fly-bys
    # Vinf_dep:         dv leaving initial planet (km/s)
    # e_target:         insertion orbit eccentricity
    # tof_bounds:       window for entire trajectory - days
    # n:                number of iterations
    # t_step:           time step (days)

    # Common parameters
    target_satellite = titan
    target_orbit = [titan.radius + 200*1e3, 0]

    # Alpha Transcription MGA, no variation in departure time, single objective
    encoding = 'alpha'  # change 'alpha' to 'direct' if you want to use the direct method
    T0mjd2000 = T0.mjd2000
    dt_arrival = np.linspace(2000, 4000, n, True)  # 100 = 100 days post launch; 3000 = 3000 days post launch;n = # steps
    dt_departure = np.linspace(T0mjd2000 - t_step*int(n/2), T0mjd2000 + t_step*int(n/2), n, True)
    
    dVs = np.empty([n, n])
    arrival_times = np.empty([n, n])
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    
    with open(out_filename, "w", newline="") as csv_f:
        writer = DictWriter(csv_f, ["t_depart [mjd2000]", "t_arrive [mjd2000]", "DV [km/s]", "champ_inter", "champ_plan"])
        writer.writeheader()
        csv_f.flush()
        
        # Get all the sequences
        print(f"{bcolors.BOLD}{bcolors.OKCYAN}Running Porkchop ...{bcolors.ENDC}\n")

        # Write the result every time one is received
        for i, departure in enumerate(dt_departure):
            print("Departure date {} of {}".format(i + 1, n))
            for j, arrival in enumerate(dt_arrival):
                success = False
                count = 0
                while not success and count < 3:
                    try:
                        champ_inter, champ_plan = traj_solver.entire_trajectory(planet_sequence, [pk.epoch(departure), pk.epoch(departure)], target_satellite, target_orbit, [arrival, arrival], encoding)
                        dv, t_depart, t_arrive, t_phases, tof, DV_inter = trajectory.get_results(champ_inter, champ_plan)
                        dVs[i][j] = dv/1000
                        arrival_times[i][j] = t_arrive.mjd2000 + departure
                        result = {"t_depart [mjd2000]":departure, "t_arrive [mjd2000]":t_arrive.mjd2000, "DV [km/s]":dv/1000, "champ_inter":champ_inter, "champ_plan":champ_plan}
                        success = True
                    except:
                        count +=1
                        result = {"t_depart [mjd2000]":departure, "t_arrive [mjd2000]":t_arrive.mjd2000, "DV [km/s]":-1, "champ_inter":None, "champ_plan":None}

                writer.writerow(result)
                csv_f.flush()

    # Plot
    # X,Y = np.meshgrid(dt_departure, dt_arrival)

    # fig1 = plt.figure(figsize=(12, 8))
    # plt.contourf(X, Y, dVs, 10)
    # plt.xlabel("Earth Launch Date")
    # plt.ylabel("Titan Arrival Date")
    # #plt.title("Earth to Titan Porkchop Plot")
    # clb = plt.colorbar()
    # clb.set_label(r'$\Delta V$ [km/s]')

    # plt.show()

    return

'''
#######################################################################################################################
Calling and Using the Porkchop Plotter Function
    - It is here where the user should change parameters for the output they expect
#######################################################################################################################
'''

if __name__ == "__main__":
    spice_kernels()
    venus, earth, mars, jupiter, saturn, titan = load_spice()

    T0 = pk.epoch_from_string("2031-Apr-22 01:07:58.334417")           # Launch date
    planet_sequence = [earth, mars, venus, earth, jupiter, saturn]   # Start at Earth, end at Saturn
    Vinf_dep = 4.                 # km/s, dv leaving initial planet
    e_target = 0.75               # orbit insertion eccentricity, ND
    T0_u = 0                      # upper bound on departure time
    tof_bounds = [50, 6000]       # window for entire trajectory - days
    n = 10                        # number of iterations
    t_step = 18.0                 # time step - days
    sequence_string = "".join([planet.name[0] for planet in planet_sequence])
    
    csv_filename = "results/porkchop_{}.csv".format(sequence_string)

    # put the call to the function in a loop so we create multiple porkchop plots at once
    # for i in porkchop_plotter:

    porkchop_plotter(trajectory, T0, planet_sequence, Vinf_dep, e_target, tof_bounds, n, t_step, csv_filename)