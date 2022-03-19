from multiprocessing import Pool, cpu_count, Process
from csv import DictWriter
import numpy as np
import os
from chemical_propulsion2 import TitanChemicalUDP
from planetary_system import PlanetToSatellite
from trajectory_solver import TrajectorySolver, load_spice, spice_kernels
from itertools import repeat
from datetime import datetime as dt
import pykep as pk
import pickle
from spice_interpolation import new_planet

"""
Two main functions:
1) Function that runs the trajectory and spits out the things we care about
2) Function that writes to the csv 
"""

# Instantiate the class once from the start and don't do it again
trajectory = TrajectorySolver(TitanChemicalUDP, PlanetToSatellite)

def traj_analysis(args):
    """
    Solve a single trajectory and output the results.

    Args:
        args (``array``) : contains all the arguments needed to run the trajectory solver and also contains the case number
    """
    
    sequence, departure_dates, target_satellite, target_orbit, case = args
    champ_inter, champ_plan = trajectory.entire_trajectory(sequence, departure_dates, target_satellite, target_orbit)
    DV, t_depart, t_arrive, t_phases, tof = trajectory.get_results(champ_inter, champ_plan)
    
    data = {"case_no":case+1, "sequence":"".join([planet.name[0] for planet in sequence]), "total_DV":DV, "t_depart":t_depart, "t_arrive":t_arrive,
                                    "tof":tof, "t_phases":t_phases, "champ_inter":list(champ_inter), "champ_plan":list(champ_plan)}
    
    return data

def load_interp(filepath):
    """
    Loads an interpolated planet object

    Args:
        filepath (``string``): the path to the planet object
    """
    with open(filepath, "rb") as inp:
            planet_new = pickle.load(inp)
    return planet_new

def extract_seqs(doe_filename, planet_dic, add_start_end=False, starting=None, ending=None):
    """
    Extract the sequences from a csv where the first column has the sequence as a number, i.e. 111.

    Args:
        doe_filename (``str``): the filename of the csv with all the sequences
        planet_dic (``dic(int:pykep.planet)``): a dictionary mapping what planet the numbers in the sequence correspond to, e.g. 1 : EARTH
    """
    
    # DOE defined as one column with just the pattern
    rows = []
    with open(doe_filename) as f:
        for row in f[1:]: # Ignore the header column
            rows.append(row.split()[0])
    
    sequences = [] # Can change this to a defined list of size _
    for row in rows:
        seq = [planet_dic[int(planet)] for planet in str(row) if planet_dic[int(planet)] != None]
        if add_start_end: # adding the starting element
            seq = [starting] + seq + [ending]
        sequences.append(seq)
    
    return sequences
    
def main(doe_filename, planet_dic, out_filename, departure_window, target_satellite, target_orbit, 
         start, append_seq=False, start_seq=None, end_seq=None):
    
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    
    with open(out_filename, "w", newline="") as csv_f:
        writer = DictWriter(csv_f, ["case_no","sequence","total_DV","t_depart","t_arrive",
                                    "tof","t_phases","champ_inter","champ_plan"])
        writer.writeheader() 
        
        # Get all the sequences
        sequences = [[earth, venus, saturn]]#extract_seqs(doe_filename, planet_dic, append_seq, start_seq, end_seq)
        cases = len(sequences)

        all_cases = list(zip(sequences, repeat(departure_window), repeat(target_satellite), 
                             repeat(target_orbit), range(cases)))
        
        p = Pool(processes=cpu_count()-2)
        results = p.imap_unordered(traj_analysis, all_cases)
        p.close()

        for i, result in enumerate(results):
            if i % 100 == 0:
                print("elapsed time:",dt.now() - start)
                print("Got result #{} ({:.2f}%)".format(i, i*100/cases))
            writer.writerow(result)

if __name__ == "__main__":
    #spice_kernels()
    #venus, earth, mars, jupiter, saturn, titan = load_spice()
    venus = load_interp("planets/VENUS1990-2050_1.pkl")
    earth = load_interp("planets/EARTH1990-2050_0.1.pkl")
    mars = load_interp("planets/MARS1990-2050_1.pkl")
    jupiter = load_interp("planets/JUPITER1990-2050_1.pkl")
    saturn = load_interp("planets/SATURN1990-2050_0.1.pkl")
    titan = load_interp("planets/TITAN1990-2050_0.1.pkl")
    print("Imported Interpolations...")
    
    start = dt.now()
    input_filename = ""
    output_filename = "../results/AEON_" + dt.date(start).isoformat() + ".csv"
    planet_dic = {1:earth, 2:venus, 3:mars, 4:jupiter, 5:None}
    departure_window = [] #### NEED TO FIX THIS EVENTUALLY
    target = titan
    target_orbit = [titan.radius * 2, 0.1]
    
    x = traj_analysis([[earth,saturn], departure_window, titan, target_orbit, 1])
    print(x)
    
    # main(input_filename, planet_dic, output_filename, departure_window, target, target_orbit, 
    #     start, append_seq=True, start_seq=earth, end_seq=saturn)