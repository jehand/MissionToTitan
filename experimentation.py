import os
from multiprocessing import Pool, cpu_count, set_start_method
from csv import DictWriter, DictReader
from udps.chemical_propulsion_mk import TitanChemicalUDP
from udps.planetary_system import PlanetToSatellite
from trajectory_solver import TrajectorySolver, load_spice, spice_kernels
from itertools import repeat
from datetime import datetime as dt
from display_style import bcolors
from ast import literal_eval
import pygmo as pg
from fcmaes import advretry
from fcmaes.optimizer import single_objective, dtime, logger, de_cma
import math
from pykep.planet import jpl_lp

# Just testing out fcmaes
def test_problem(opt, problem, num_retries = 10000, num = 3, value_limit = math.inf, log = logger()):
    log.info(problem.name + ' ' + opt.name)
    for _ in range(num):
        _ = advretry.minimize(problem.fun, problem.bounds, value_limit, num_retries, log, optimizer=opt)
        

class newUDP(TitanChemicalUDP):
    def __init__(self, sequence):
        
        spice_kernels()
        super().__init__(
            sequence=sequence,
            constrained=False
        )
        
        self.sequence = sequence
       
if __name__ == "__main__":        
    spice_kernels()
    venus, earth, mars, jupiter, saturn, titan = load_spice()
    
    planetary_sequence = [earth,venus,venus,earth,jupiter,saturn]
    planetary_sequence = [jpl_lp("earth"), jpl_lp("venus"), jpl_lp("venus"), jpl_lp("earth"), jpl_lp("jupiter"), jpl_lp("saturn")]
    udp = TitanChemicalUDP(sequence=planetary_sequence, constrained=False)
    #udp = newUDP(planetary_sequence)
    
    # prob = single_objective(pg.problem(udp))
    # opt = de_cma(de_max_evals = 750, cma_max_evals = 750)
    # best = test_problem(opt, prob, num_retries = 1000, value_limit = 100000)
    # print(best)
    
    champion = [-1020.997299028031, 0.22617406116388966, 0.5203025355408949, 4154.105109794489, 0.23101194142993417, 0.2737902412835537, 1.5939976375824945, 4.332270785331538, 0.30608328477022395, 0.08452274974947302, -1.6662501651614154, 1.4710789969567075, 0.011342192803034671, 0.18387830609206396, -1.4145381120924236, 1.1001885889420064, 0.039926507884149774, 0.24637273238055404, -4.589812321666531, 255.52616478674136, 0.02390697876731314, 0.9903480472513908]
    udp.pretty(champion)