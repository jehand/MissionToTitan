import os
from multiprocessing import Pool, cpu_count, set_start_method
from csv import DictWriter, DictReader
from udps.chemical_mga import TitanChemicalMGAUDP
from udps.planetary_system import PlanetToSatellite
from trajectory_solver import TrajectorySolver, load_spice, spice_kernels
from itertools import repeat
from datetime import datetime as dt
from display_style import bcolors
from ast import literal_eval
import pygmo as pg
from datetime import datetime as dt
import numpy as np

global_algo_dic = {"SADE": pg.sade(gen=500, ftol=1e-10, xtol=1e-10),
                   "DE": pg.de(gen=500, ftol=1e-10, xtol=1e-10),
                   "GACO": pg.gaco(gen=500),
                   "DE_1220": pg.de1220(gen=500, ftol=1e-10, xtol=1e-10),
                   "GWO": pg.gwo(gen=500),
                   "IHS": pg.ihs(gen=500),
                   "PSO": pg.pso(),
                   "GPSO": pg.pso_gen(gen=500),
                   "SEA": pg.sea(),
                   "SGA": pg.sga(),
                   "SA": pg.simulated_annealing(),
                   "ABC": pg.bee_colony(),
                   "CMA-ES": pg.cmaes(),
                   "xNES": pg.xnes(gen=500,ftol=1e-10,xtol=1e-10), 
                   "NSGA2": pg.nsga2(),
                   "MOEA/D": pg.moead(),
                   "MHACO": pg.maco(),
                   "NSPSO": pg.nspso()
                   }
                   
local_algo_dic = {"COMPASS": pg.compass_search(),
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

def _evolve_func(algo, pop): # doctest : +SKIP
    new_pop = algo.evolve(pop)
    return algo, new_pop
class mp_island(pg.mp_island): # doctest : +SKIP
    def __init__(self):
        # Init the process pool, if necessary.
        self._use_pool = True
        # mp_island.shutdown_pool()
        mp_island.init_pool()
        mp_island._pool.apply(spice_kernels)

    def run_evolve(self, algo, pop):
        with mp_island._pool_lock:
            res = mp_island._pool.apply_async(_evolve_func, (algo, pop))
        return res.get()

class topo:
    def __init__(self, n_islands):
        self.n_islands = n_islands
        self.connections = {}
        self.create_connections(n_islands)

    def get_connections(self, n):
        return self.connections[n]

    def push_back(self):
        return

    def get_name(self):
        return "Rim Topology"
    
    def create_connections(self, n_islands):
        for i in range(n_islands):
            if i==0:
                connection = list(range(1, n_islands))
                self.connections[i] = [connection,np.ones(len(connection))]
            else:
                self.connections[i] = [[i-1, i+1, 0], [1,1,1]]
        return

def analysis(args):
    udp, algos, case = args
    spice_kernels()
    
    pop_size = 32
    try:
        start_time = dt.now()
        alg_glob_no_mbh = pg.algorithm(global_algo_dic[algos[0]])
        alg_glob = pg.algorithm(pg.mbh(algo=alg_glob_no_mbh, stop=3, perturb=.25))
        alg_loc = pg.algorithm(local_algo_dic[algos[1]])
                
        isls = [pg.island(algo = alg_loc, prob=udp, size=pop_size, udi=mp_island())]

        for _ in range(6):
            isls.append(pg.island(algo=alg_glob, prob=udp, size=pop_size, udi=mp_island()))

        archi = pg.archipelago()
        for isl in isls:
            archi.push_back(isl)

        archi.set_topology(topo(n_islands=len(isls)))
        #print("Evolving Archipelago...")
        archi.evolve(3)
        archi.wait()

        sols = archi.get_champions_f()
        idx = sols.index(min(sols))
        champion = archi.get_champions_x()[idx]
        DV = sols[idx][0]
        T = udp._compute_dvs(champion)[5]
        tof = champion[0] + sum(T)
        time_taken = (dt.now() - start_time).total_seconds()

        data = {"case_no":case+1,"global_algo":algos[0],"local_algo":algos[1],"DV":DV,"tof":tof,"time_taken":time_taken,"champion":champion}
    except Exception as e:
        data = {"case_no":case+1,"global_algo":algos[0],"local_algo":algos[1],"DV":"FAILED","tof":None,"time_taken":None,"champion":None}
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
        writer = DictWriter(csv_f, ["case_no","global_algo","local_algo","DV","tof","time_taken","champion"])
        writer.writeheader()
        
        # Get all the sequences
        algo_combinations = algorithm_combinations(global_algos, local_algos)
        cases = len(algo_combinations)
        print(f"{bcolors.ITALIC}{bcolors.WARNING}Found {cases} Algorithm Combinations ...{bcolors.ENDC}\n")

        # Compile all the cases
        all_cases = list(zip(repeat(udp), algo_combinations, range(cases)))

        # Write the result every time one is received
        for i, case in enumerate(all_cases):
            result = analysis(case)
            if result["DV"] == "FAILED":
                color = bcolors.FAIL
                success = "FAIL"
            else:
                color = bcolors.OKGREEN
                success = "SUCCESS"
            print("{}{}! Got result #{} ({:.2f}%){}".format(color, success, i+1, (i+1)*100/cases, bcolors.ENDC))
            print(f"\t{bcolors.ITALIC}{bcolors.SUBTITLE}Elapsed time: {dt.now() - start} \n {bcolors.ENDC}")
            writer.writerow(result)
            csv_f.flush()

if __name__ == "__main__":
    set_start_method("spawn")
    spice_kernels()
    venus, earth, mars, jupiter, saturn, titan = load_spice()
    
    global_algos = list(global_algo_dic.keys()) #["SADE", "DE", "GACO"]
    local_algos = ["COMPASS", "NLOPT-BOBYQA", "NLOPT-PRAXIS", "NLOPT-COBYLA", "NLOPT-SLSQP", "NLOPT-MMA"]
    output_filename = "results/archi_algorithm_comparison3.csv"
    
    # Testing the analysis function
    planetary_sequence = [earth,venus,venus,earth,jupiter,saturn]
    udp = TitanChemicalMGAUDP(sequence=planetary_sequence, constrained=False)
    
    # Run all algos with python multiprocessing
    main(udp=udp, global_algos=global_algos, local_algos=local_algos, out_filename=output_filename)