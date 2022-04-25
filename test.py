import os
from multiprocessing import Pool, cpu_count, set_start_method
from csv import DictWriter, DictReader
from tkinter import E
from udps.chemical_propulsion_mk import TitanChemicalUDP
from udps.chemical_mga import TitanChemicalMGAUDP
from udps.planetary_system import PlanetToSatellite
from trajectory_solver import TrajectorySolver, load_spice, spice_kernels
from itertools import repeat
from datetime import datetime as dt
from display_style import bcolors
from ast import literal_eval
import pygmo as pg
from pykep.planet import jpl_lp
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pykep as pk
from pykep.trajopt import mga

global_algo_dic = {"SADE": pg.sade(gen=5000, ftol=1e-10, xtol=1e-10),
                   "DE": pg.de(),
                   "GACO": pg.gaco(gen=5000),
                   "DE_1220": pg.de1220(gen=5000, ftol=1e-10, xtol=1e-10),
                   "GWO": pg.gwo(gen=5000),
                   "IHS": pg.ihs(gen=5000),
                   "PSO": pg.pso(),
                   "GPSO": pg.pso_gen(gen=5000),
                   "SEA": pg.sea(),
                   "SGA": pg.sga(),
                   "SA": pg.simulated_annealing(),
                   "ABC": pg.bee_colony(),
                   "CMA-ES": pg.cmaes(),
                   "xNES": pg.xnes(gen=5000,ftol=1e-10,xtol=1e-10), 
                   "NSGA2": pg.nsga2(),
                   "MOEA/D": pg.moead(),
                   "MHACO": pg.maco(),
                   "NSPSO": pg.nspso()
                   }
                   
local_algo_dic = {"COMPASS": pg.compass_search(max_fevals=1000, start_range=1e-2, stop_range=1e-5, reduction_coeff=0.5),
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

class my_isl:
    def run_evolve(self, algo, pop):
        new_pop = algo.evolve(pop)
        return algo, new_pop
    def get_name(self):
        return "It's my island!"

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

def eta_final_epoch(x, tof_max=3000):
    t0 = x[0]
    etas = x[1:]
    T = []
    for eta in etas:
        T.append((tof_max - np.sum(T))*eta)

    return t0 + sum(T)

def testing_trajectory(t0, t0fmax, leg_times):
    # Decision chromosome: [t0, n1, n2, n3, ...]
    x = [t0]
    T = []

    for leg in leg_times:
        eta = (leg)/(t0fmax - np.sum(T))
        x.append(eta)
        T.append(leg)

    return x

if __name__ == "__main__":
    spice_kernels()
    venus, earth, mars, jupiter, saturn, titan = [jpl_lp("venus"), jpl_lp("earth"), jpl_lp("mars"), jpl_lp("jupiter"), jpl_lp("saturn"), None]
    #venus, earth, mars, jupiter, saturn, titan = load_spice() 

    # Testing the analysis function
    #planetary_sequence = [earth, earth, venus, earth, earth, saturn]#[earth,venus,venus,earth,jupiter,saturn]
    planetary_sequence = [earth, venus, venus, earth, jupiter, saturn]
    #udp = TitanChemicalUDP(sequence=planetary_sequence)

    # Making a new mga function
    # udp = mga(
    #             seq=planetary_sequence,
    #             t0=[pk.epoch_from_string("1997-JAN-01 00:00:00.000"), pk.epoch_from_string("1997-DEC-31 00:00:00.000")],
    #             tof=3000,
    #             vinf=4.25,
    #             tof_encoding='eta',
    #             multi_objective=False,
    #             orbit_insertion=True,
    #             e_target=.9823,
    #             rp_target=78232 * 1e3,
    #             max_revs= 3,
    #         )

    udp = TitanChemicalMGAUDP(sequence=planetary_sequence, tof=[3000,3000], tof_encoding="alpha")

    # Defining the algo
    start_time = dt.now()
    pop_size = 32
    isls = [pg.island(algo = local_algo_dic["COMPASS"], prob=udp, size=pop_size, udi=mp_island())]

    # variants = [5,1,2,5,1,2]
    # for var in variants:
    #     algorithm = pg.algorithm(pg.de(gen=500, variant=var, ftol=1e-10, xtol=1e-10))
    #     isls.append(pg.island(algo=pg.mbh(algo=algorithm, stop=5, perturb=0.25), prob=udp, size=pop_size))

    # Trying DE1220
    variant_adptvs = [1,2] * 3 #int(np.floor(cpu_count()/2))

    allowed_variants = list(range(1,19))
    for var in variant_adptvs:
        algorithm = pg.algorithm(pg.de1220(gen=3, variant_adptv=var, ftol=1e-10, xtol=1e-10))
        isls.append(pg.island(algo=pg.mbh(algo=algorithm, stop=3, perturb=0.25), prob=udp, size=pop_size, udi=mp_island()))

    archi = pg.archipelago()
    for isl in isls:
        archi.push_back(isl)

    archi.set_topology(topo(n_islands=len(isls)))
    print("Evolving Archipelago...")
    archi.evolve(3)
    archi.wait()
        
    sols = archi.get_champions_f()
    idx = sols.index(min(sols))
    end_time = dt.now()
    best_x = archi.get_champions_x()[idx]
    print(best_x)

    # t0 = pk.epoch_from_string("2025-APR-12 00:00:00.000").mjd2000
    # times = [t0,
    #          pk.epoch_from_string("2026-APR-11 00:00:00.000").mjd2000, 
    #          pk.epoch_from_string("2027-APR-16 00:00:00.000").mjd2000,
    #          pk.epoch_from_string("2028-MAY-27 00:00:00.000").mjd2000,
    #          pk.epoch_from_string("2031-SEP-03 00:00:00.000").mjd2000,
    #          pk.epoch_from_string("2031-DEC-02 00:00:00.000").mjd2000
    #          ]

    # t0 = pk.epoch_from_string("1997-OCT-6 00:00:00.000").mjd2000
    # times = [t0,
    #          pk.epoch_from_string("1998-APR-21 00:00:00.000").mjd2000, 
    #          pk.epoch_from_string("1999-JUN-20 00:00:00.000").mjd2000,
    #          pk.epoch_from_string("1999-AUG-16 00:00:00.000").mjd2000,
    #          pk.epoch_from_string("2000-DEC-30 00:00:00.000").mjd2000,
    #          pk.epoch_from_string("2004-JUL-01 00:00:00.000").mjd2000
    #          ]


    # times_per_leg = np.diff(times)
    # best_x = testing_trajectory(t0, 3000, times_per_leg)    
    udp.pretty(best_x)
    udp.plot(best_x)
    plt.show()