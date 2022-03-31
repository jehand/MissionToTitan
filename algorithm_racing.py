import os
from multiprocessing import Pool, cpu_count, set_start_method
from csv import DictWriter, DictReader
from udps.chemical_propulsion2 import TitanChemicalUDP
from udps.planetary_system import PlanetToSatellite
from trajectory_solver import TrajectorySolver, load_spice, spice_kernels
from itertools import repeat
from datetime import datetime as dt
from display_style import bcolors
from ast import literal_eval
import pygmo as pg

global_algo_dic = {"SADE": pg.sade(),
                   "DE": pg.de(),
                   "GACO": pg.gaco(),
                   "DE_1220": pg.de1220(),
                   "GWO": pg.gwo(),
                   "IHS": pg.ihs(),
                   "PSO": pg.pso(),
                   "GPSO": pg.pso_gen(),
                   "SEA": pg.sea(),
                   "SGA": pg.sga(),
                   "SA": pg.simulated_annealing(),
                   "ABC": pg.bee_colony(),
                   "CMA-ES": pg.cmaes(),
                   "xNES": pg.xnes(),
                   "NSGA2": pg.nsga2(),
                   "MOEA/D": pg.moead(),
                   "MHACO": pg.maco(),
                   "NSPSO": pg.nspso()
                   }
local_algo_dic = {"Compass": pg.compass_search(),
                  "NLOPT-COBYLA": pg.nlopt("cobyla"),
                  "NLOPT-BOBYQA": pg.nlopt("bobyqa"),
                  "NLOPT-NEWUOA": pg.nlopt("newuoa"),
                  "NLOPT-NEWUOA_BOUND": pg.nlopt("newuoa_bound"),
                  "NLOPT-PRAXIS": pg.nlopt("praxis"),
                  "NLOPT-NELDERMEAD": pg.nlopt("neldermead"),
                  "NLOPT-SBPLX": pg.nlopt("sbplx"),
                  "NLOPT-MMA": pg.nlopt("mma"),
                  "NLOPT-CCSAQ": pg.nlopt("ccsaq"),
                  "NLOPT-SLSQP": pg.nlopt("slsqp"),
                  "NLOPT-LBFGS": pg.nlopt("lbfgs"),
                  "NLOPT-TNEWTON_PRECOND_RESTART": pg.nlopt("tnewton_precond_restart"),
                  "NLOPT-TNEWTON_PRECOND": pg.nlopt("tnewton_precond"),
                  "NLOPT-TNEWTON_RESTART": pg.nlopt("tnewton_restart"),
                  "NLOPT-TNEWTON": pg.nlopt("tnewton"),
                  "NLOPT-VAR2": pg.nlopt("var2"),    
                  "NLOPT-VAR1": pg.nlopt("var1"),
                  "NLOPT-IPOPT": pg.ipopt(),
                  "SCIPY-NELDERMEAD": pg.scipy_optimize(method="Nelder-Mead"),
                  "SCIPY-POWELL": pg.scipy_optimize(method="Powell"),
                  "SCIPY-CG": pg.scipy_optimize(method="CG"),
                  "SCIPY-BFGS": pg.scipy_optimize(method="BFGS"),
                  "SCIPY-NEWTON-CG": pg.scipy_optimize(method="Newton-CG"),
                  "SCIPY-L-BFGS-B": pg.scipy_optimize(method="L-BFGS-B"),
                  "SCIPY-TNC": pg.scipy_optimize(method="TNC"),
                  "SCIPY-COBYLA": pg.scipy_optimize(method="COBYLA"),
                  "SCIPY-SLSQP": pg.scipy_optimize(method="SLSQP"),
                  "SCIPY-TRUST-CONSTR": pg.scipy_optimize(method="trust-constr"),
                  "SCIPY-DOGLEG": pg.scipy_optimize(method="dogleg"),
                  "SCIPY-TRUST-NCG": pg.scipy_optimize(method="trust-ncg"),
                  "SCIPY-TRUST-EXACT": pg.scipy_optimize(method="trust-exact"),
                  "SCIPY-TRUST-KRYLOV": pg.scipy_optimize(method="trust-krylov")
                  }

def analysis(args):
    udp, algos, case = args
    spice_kernels()
    
    pop_n = 1000
    pop = pg.population(udp, pop_n)
    try:
        alg_glob = pg.algorithm(global_algo_dic[algos[0]])
        alg_loc = pg.algorithm(local_algo_dic[algos[1]])
        pop = alg_glob.evolve(pop)
        pop = alg_loc.evolve(pop)

        champion = pop.champion_x
        DV, _, T, _, _ = udp._compute_dvs(champion)
        tof = sum(T)
        DV = sum(DV)
        
        data = {"case_no":case+1,"global_algo":algos[0],"local_algo":algos[1],"DV":DV,"tof":tof,"champion":champion}
    except Exception as e:
        data = {"case_no":case+1,"global_algo":algos[0],"local_algo":algos[1],"DV":"FAILED","tof":None,"champion":None}
        
    return data
    

def algorithm_combinations(global_algos, local_algos):
    all_combinations = []    
    for glob in global_algos:
        for local in local_algos:
            all_combinations.append([glob, local])
            
    return all_combinations

def main(udp, global_algos, local_algos, out_filename):
    start = dt.now()
    print(f"{bcolors.BOLD}{bcolors.WARNING}Initializing Full Factorial on the Following Algorithms ...{bcolors.ENDC}")
    
    print("\t{}{}Global Algorithms: {}{}".format(bcolors.ITALIC, bcolors.SUBTITLE, global_algos, bcolors.ENDC))
    print("\t{}{}Local Algorithms: {}{}".format(bcolors.ITALIC, bcolors.SUBTITLE, local_algos, bcolors.ENDC))
    
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    with open(out_filename, "w", newline="") as csv_f:
        writer = DictWriter(csv_f, ["case_no","global_algo","local_algo","DV","tof","champion"])
        writer.writeheader()
        
        # Get all the sequences
        algo_combinations = algorithm_combinations(global_algos, local_algos)
        cases = len(algo_combinations)
        print(f"{bcolors.ITALIC}{bcolors.WARNING}Found {cases} Algorithm Combinations ...{bcolors.ENDC}\n")

        # Compile all the cases
        all_cases = list(zip(repeat(udp), algo_combinations, range(cases)))
        
        # Set up the multiprocessor and run
        print(f"{bcolors.BOLD}{bcolors.OKCYAN}Running Full Factorial ...{bcolors.ENDC}\n")
        p = Pool(processes=cpu_count()-2)
        results = p.imap_unordered(analysis, all_cases)
        p.close()

        # Write the result every time one is received
        for i, result in enumerate(results):
            if result["DV"] == "FAILED":
                color = bcolors.FAIL
                success = "FAIL"
            else:
                color = bcolors.OKGREEN
                success = "SUCCESS"
            print("{}{}! Got result DV = {:.2f}km/s ({:.2f}%){}".format(color, success, result["DV"]/1000, (i+1)*100/cases, bcolors.ENDC))
            print(f"\t{bcolors.ITALIC}{bcolors.SUBTITLE}Elapsed time: {dt.now() - start} \n {bcolors.ENDC}")
            writer.writerow(result)

if __name__ == "__main__":
    set_start_method("spawn")
    spice_kernels()
    venus, earth, mars, jupiter, saturn, titan = load_spice()
    
    global_algos = ["SADE", "DE", "GACO"]
    local_algos = ["Compass"]
    output_filename = "results/algorithm_comparison.csv"
    
    # Testing the analysis function
    planetary_sequence = [earth,venus,earth,mars,jupiter,saturn]
    udp = TitanChemicalUDP(sequence=planetary_sequence, constrained=False)
    
    # Run all algos with python multiprocessing
    main(udp=udp, global_algos=global_algos, local_algos=local_algos, out_filename=output_filename)