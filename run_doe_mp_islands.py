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
#from mpi4py import MPI

# Instantiate the class once from the start and don't do it again
interplanetary_udp = TitanChemicalMGAUDP
planetary_udp = PlanetToSatellite
trajectory = TrajectorySolver(interplanetary_udp, planetary_udp)

# Instantiate MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# nprocs = comm.Get_size()

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def traj_analysis(args):
    """
    Solve a single trajectory and output the results.

    Args:
        args (``array``) : contains all the arguments needed to run the trajectory solver and also contains the case number
    """
    sequence, departure_dates, target_satellite, target_orbit, case = args
    
    try:
        champ_inter, champ_plan = trajectory.entire_trajectory(sequence, departure_dates, target_satellite, target_orbit)
        DV, t_depart, t_arrive, t_phases, tof, DV_inter = trajectory.get_results(champ_inter, champ_plan)

        data = {"case_no":case+1, "sequence":"".join([planet.name[0] for planet in sequence]), "total_DV":DV, "DV_inter":DV_inter, "DV_plan":DV-DV_inter, "t_depart":t_depart, 
                "t_arrive":t_arrive, "tof":tof, "t_inter":t_phases[0], "t_plan":t_phases[1], "champ_inter":list(champ_inter), "champ_plan":list(champ_plan)}
    except Exception as e:
        data = {"case_no":case+1, "sequence":"".join([planet.name[0] for planet in sequence]), "total_DV":"FAILED", "DV_inter":None, "DV_plan":None, "t_depart":None, 
                "t_arrive":None, "tof":None, "t_inter":None, "t_plan":None, "champ_inter":None, "champ_plan":None}
        
    return data

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
        for row in f.readlines()[1:]: # Ignore the header column
            rows.append(row.split(",")[0])

    sequences = [] # Can change this to a defined list of size _
    for row in rows:
        seq = [planet_dic[int(planet)] for planet in str(row) if planet_dic[int(planet)] != None]
        if add_start_end: # adding the starting element
            seq = [starting] + seq + [ending]
        if seq not in sequences:
            sequences.append(seq)
    
    return sequences
    
def main(doe_filename, planet_dic, out_filename, departure_window, target_satellite, target_orbit, 
         start, append_seq=False, start_seq=None, end_seq=None):
    
    print(f"{bcolors.BOLD}{bcolors.WARNING}Initializing DoE ...{bcolors.ENDC}")
    
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    
    with open(out_filename, "w", newline="") as csv_f:
        writer = DictWriter(csv_f, ["case_no","sequence","total_DV","DV_inter","DV_plan","t_depart","t_arrive",
                                    "tof","t_inter","t_plan","champ_inter","champ_plan"])
        writer.writeheader()
        csv_f.flush()
        
        # Get all the sequences
        sequences = extract_seqs(doe_filename, planet_dic, append_seq, start_seq, end_seq)
        cases = len(sequences)
        print(f"\t{bcolors.ITALIC}{bcolors.WARNING}Found {cases} Sequences ...{bcolors.ENDC}\n")

        # Compile all the cases
        all_cases = list(zip(sequences, repeat(departure_window), repeat(target_satellite), 
                             repeat(target_orbit), range(cases)))
        
        # Set up MPI and run
        #all_cases = list(split(all_cases, nprocs))
        #all_cases = comm.scatter(all_cases, root=0)
        #print('Process {} has data:'.format(rank), all_cases)
        print(f"{bcolors.BOLD}{bcolors.OKCYAN}Running DoE ...{bcolors.ENDC}\n")

        # Write the result every time one is received
        for i, case in enumerate(all_cases):
            result = traj_analysis(case)
            if result["total_DV"] == "FAILED":
                color = bcolors.FAIL
                success = "FAIL"
            else:
                color = bcolors.OKGREEN
                success = "SUCCESS"
            print("{}{}! Got result #{} ({:.2f}%){}".format(color, success, i+1, (i+1)*100/cases, bcolors.ENDC))
            print(f"\t{bcolors.ITALIC}{bcolors.SUBTITLE}Elapsed time: {dt.now() - start} \n {bcolors.ENDC}")
            writer.writerow(result)
            csv_f.flush()

def read_doe_case(doe_filename, sequence):
    """
    Reads in a specific case in the DoE output, defined using the sequence, and outputs the row as a dictionary

    Args:
        doe_filename (``str``): the filepath to the output DoE file
        sequence (``str``): description of the sequence, e.g. 'EVEMJS' 
    """
    with open(doe_filename) as f:
        reader = DictReader(f)
        case = next((item for item in reader if item["sequence"] == sequence), None)
    
    # Give an error if the sequence isn't found
    if case is None:
        raise LookupError("ERROR: Sequence not found in file")
    
    # Convert everything to the correct dtype
    case['case_no'] = int(case["case_no"])
    case["total_DV"] = float(case["total_DV"])
    case["t_depart"] = float(case["t_depart"])
    case["t_arrive"] = float(case["t_arrive"])
    case["tof"] = float(case["tof"])
    case["t_phases"] = literal_eval(case["t_phases"])
    case["champ_inter"] = literal_eval(case["champ_inter"])
    case["champ_plan"] = literal_eval(case["champ_plan"])
    
    return case

def pretty_doe_result(doe_filename, sequence, planets, target, target_orbit):
    """
    Prints the pretty results for a single trajectory read in from a DoE output file

    Args:
        doe_filename (``str``): the filepath to the output DoE file
        sequence (``str``): description of the sequence, e.g. 'EVEMJS' 
        planets (``array(pykep.planet)``): the array of pykep.planet objects for the interplanetary planets
        target (``pykep.planet``): the target satellite for the planetary stage
    """
    case = read_doe_case(doe_filename, sequence)
    champion_interplanetary = case["champ_inter"]
    champion_planetary = case["champ_plan"]
    
    n = len(sequence)
    planetary_sequence = []
    for planetary_letter in list(sequence):
        for planet in planets:
            if planetary_letter == planet.name[0]:
                planetary_sequence.append(planet)
    
    trajectory.interplanetary_udp = interplanetary_udp(planetary_sequence)
    v_sc, start_time = trajectory.compute_final_vinf(planetary_sequence[-1], champion_interplanetary)
    trajectory.planetary_udp = planetary_udp(start_time, target_orbit[0], target_orbit[1], planetary_sequence[-1], target, 
                                          tof=[10,100], r_start_max=15, initial_insertion=True, v_inf=v_sc, max_revs=5)
    
    trajectory.pretty(champion_interplanetary, champion_planetary)
    
def plot_doe_result(doe_filename, sequence, planets, target, target_orbit):
    """
    Plots the results for a single trajectory read in from a DoE output file

    Args:
        doe_filename (``str``): the filepath to the output DoE file
        sequence (``str``): description of the sequence, e.g. 'EVEMJS' 
        planets (``array(pykep.planet)``): the array of pykep.planet objects for the interplanetary planets
        target (``pykep.planet``): the target satellite for the planetary stage
    """
    case = read_doe_case(doe_filename, sequence)
    champion_interplanetary = case["champ_inter"]
    champion_planetary = case["champ_plan"]
    
    n = len(sequence)
    planetary_sequence = []
    for planetary_letter in list(sequence):
        for planet in planets:
            if planetary_letter == planet.name[0]:
                planetary_sequence.append(planet)
                
    trajectory.interplanetary_udp = interplanetary_udp(planetary_sequence)
    v_sc, start_time = trajectory.compute_final_vinf(planetary_sequence[-1], champion_interplanetary)
    trajectory.planetary_udp = planetary_udp(start_time, target_orbit[0], target_orbit[1], planetary_sequence[-1], target, 
                                          tof=[10,100], r_start_max=15, initial_insertion=True, v_inf=v_sc, max_revs=5)
    
    trajectory.plot(champion_interplanetary, champion_planetary)


if __name__ == "__main__":
    set_start_method("spawn")
    spice_kernels()
    #titan = load_spice()[-1]
    #venus, earth, mars, jupiter, saturn, _ = [jpl_lp("venus"), jpl_lp("earth"), jpl_lp("mars"), jpl_lp("jupiter"), jpl_lp("saturn"), None]
    venus, earth, mars, jupiter, saturn, titan = load_spice()
    
    start = dt.now()
    input_filename = "Planetary_factorial2.csv"
    output_filename = "results/AEON_" + start.strftime("%Y-%m-%d-%H-%M-%S") + ".csv"
    planet_dic = {1:earth, 2:venus, 3:mars, 5:jupiter, 4:None}
    departure_window = [pk.epoch_from_string("2030-JAN-01 00:00:00.000"), pk.epoch_from_string("2032-DEC-31 00:00:00.000")]
    target = titan
    target_orbit = [titan.radius + 200*1e3, 0]
    
    main(input_filename, planet_dic, output_filename, departure_window, target, target_orbit, 
        start, append_seq=True, start_seq=earth, end_seq=saturn)
    
    # pretty_doe_result(output_filename, "EEEEEEVS", [venus, earth, mars, jupiter, saturn], target, target_orbit)
    # plot_doe_result(output_filename, "EEEEEEVS", [venus, earth, mars, jupiter, saturn], target, target_orbit)

    ### ALSO NEED TO ADD THE ABILITY TO RE-RUN A CASE X TIMES TO CONFIRM THAT IT IS THE OPTIMAL RESULT
