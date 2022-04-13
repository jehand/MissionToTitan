import numpy as np
import pykep as pk
from pykep.trajopt._lambert import lambert_problem_multirev
from pykep.core import epoch, DAY2SEC, MU_SUN, lambert_problem, AU, epoch, SEC2DAY, propagate_lagrangian
import pygmo as pg
import matplotlib.pyplot as plt

# try:
#     from algorithms import Algorithms
# except:
#     import sys
#     sys.path.append(sys.path[0]+"/udps")
#     from algorithms import Algorithms
    
def norm(x):
    return np.sqrt(sum([it * it for it in x]))

class PlanetToSatellite:
    """
    Decision chromosome, x = [r_start, e_start, t_apo, tof]
    
    r_start : the starting orbit periapsis, as a fraction of the initial planets radius (r_start/r_P, where r_P is radius of initial planet)
    e_start : the starting orbit eccentricity
    t_apo   : time after apoapsis to start the lambert burn, as a ratio of the total tof
    tof     : time from apoapsis in the initial orbit to periapsis in the orbit at the destination planet in days
    """
    
    def __init__(self, starting_time, r_target, e_target, starting_planet, target_planet, tof, r_start_max, 
                 initial_insertion=True, v_inf=[0,0,0], max_revs=0):
        """
        Initializing variables for the problem.

        Args:
            starting_time (``pykep.epoch``)     : the time when you are at apoapsis of the original orbit
            r_target (``float``)                : the periapsis distance for the orbit at the target planet divided by the radius of the planet
            e_target (``float``)                : the eccentricity of the target orbit
            starting_planet (``pykep.planet``)  : the initial planet object
            target_planet (``pykep.planet``)    : the destination planet object
            tof (``array(float)``)              : range of time allowed from the starting planet insertion till the target planet insertion
                                                  [t0, tf], in days
            r_start_max (``float``)             : the maximum periapsis distance allowed for the starting orbit (as a ratio of planet radius)
            initial_insertion(``bool``)         : if true, you are also optimizing for the best initial orbit accounting for v_inf (so v_inf 
                                                  must be defined)
            v_inf (``array(float)``)            : the incoming velocity vector before the initial orbit, used to find the delta_v insertion (in m/s)
            max_revs (``int``)                  : the number of revolutions allowable for the lambert solver
        """
        
        self.starting_time = starting_time
        self.r_target = r_target
        self.e_target = e_target
        self.target_planet = target_planet
        self.tof = tof
        self.r_start_max = r_start_max
        self.starting_planet = starting_planet
        self.initial_insertion = initial_insertion
        self.v_inf = v_inf
        self.max_revs = max_revs
        
        self.initial_planet_radius = starting_planet.radius
        self.starting_safe_radius = starting_planet.safe_radius / self.initial_planet_radius
    
    def get_bounds(self):
        lb = [self.starting_safe_radius, 0, 0, self.tof[0]]
        ub = [self.r_start_max, 1, 1, self.tof[1]]
        return (lb, ub)
    
    def planet_orbital_description(self, time):
        """
        This function outputs the orbital description for a planet relative to how it was defined at a specific time.
            
        Args:
            time (``pykep.epoch``)  : the time at which the orbit parameters should be determined for the target planet
        """
        planet = self.target_planet
        
        return planet.osculating_elements(epoch(time.mjd2000, "mjd2000"))
    
    def orbit2orbit_lambert(self, x):
        """
        This function calculates the Delta V required for a specific orbit to orbit transfer (matching the orbital parameters for the spacecraft, with that of the target planet, just at a different periapsis and eccentricity).
        """
        
        r_starting = x[0] * self.initial_planet_radius
        e_starting = x[1]
        target_planet = self.target_planet
        r_target = self.r_target
        e_target = self.e_target
        starting_time = self.starting_time
        dt = x[3] * (1-x[2])
        v_inf = self.v_inf
        DV = np.empty(2 + self.initial_insertion)
        
        # Find the target planets orbit parameters
        a_P, e_P, i_P, W_P, w_P, E_P = self.planet_orbital_description(starting_time)
        
        a_sc = r_starting/(1-e_starting)
        e_sc = e_starting
        i_sc = i_P
        W_sc = W_P
        w_sc = w_P
        E_sc = 0 # starting orbit at periapsis
        
        t_apo = x[3] * x[2] #total time of flight * ratio until start of lambert, i.e. time till apoapsis normally
        
        r_i,v_i = pk.par2ic([a_sc, e_sc, i_sc, W_sc, w_sc, E_sc], target_planet.mu_central_body) # at the start of the orbit
        
        # Adding the v_inf transfer
        if self.initial_insertion:
            DV[0] = norm([a-b for a,b in zip(v_i, v_inf)])
        
        r,v = propagate_lagrangian(r_i, v_i, t_apo, target_planet.mu_central_body) # propagating to when lambert starts
            
        r_Pf, v_Pf = target_planet.eph(epoch(starting_time.mjd2000 + dt + t_apo))
        
        l = lambert_problem_multirev(v, lambert_problem(r, r_Pf, dt*DAY2SEC,
                                                                target_planet.mu_central_body, cw=False, max_revs=self.max_revs))
        
        v_end_l = l.get_v2()[0]
        v_beg_l = l.get_v1()[0]
        
        DV[-2] = norm([a-b for a,b in zip(v, v_beg_l)])
        
        # Calculating the insertion burns
        DV[-1] =  norm([a - b for a, b in zip(v_end_l, v_Pf)])
        # In this case we compute the insertion DV as a single pericenter burn
        DVper = np.sqrt(DV[-1] * DV[-1] + 2 *
                        target_planet.mu_self / r_target)
        DVper2 = np.sqrt(2 * target_planet.mu_self / r_target -
                        target_planet.mu_self / r_target * (1. - e_target))
        DV[-1] = np.abs(DVper - DVper2)
                
        return np.sum(DV), [t_apo, x[3]], l, [r,v]
    
    def fitness(self, x):
        DV, _, _, _ = self.orbit2orbit_lambert(x)
        
        return [DV, ]

    def gradient(self, x):
        return pg.estimate_gradient((lambda x: self.fitness(x), x), dx = 1e-8)

    def get_name(self):
        return "Planet to Satellite"
    
    def pretty(self, x):
        # Printing pretty :) results
        DV, times, l, eph = self.orbit2orbit_lambert(x)
        print("Starting planet orbit periapsis radius:", '{0:.4g}'.format(x[0]), f"{self.starting_planet.name} radii")
        print("Starting planet orbit eccentricity:", '{0:.4g}'.format(x[1]))
        
        print("Time from initial planet orbit insertion till lambert burn:", '{0:.4g}'.format(times[0]), "days")
        print("Duration of lambert leg:", '{0:.4g}'.format(times[1]-times[0]), "days")
        print("Planetary Time:", '{0:.4g}'.format(times[1]), "days")
        print("Planetary DV:", '{0:.4g}'.format(DV/1000), "km/s")
    
    def plot(self, x, ax=None):
        DV, times, l, eph = self.orbit2orbit_lambert(x)
        
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        else:
            ax=ax
        
        pk.orbit_plots.plot_planet(self.target_planet, t0=epoch(self.starting_time.mjd2000+times[1]), axes=ax, color="b", units=AU, legend=True)
        pk.orbit_plots.plot_lambert(l, units=AU, axes=ax, color="r", legend=True)
        pk.orbit_plots.plot_kepler(eph[0], eph[1], times[0]*DAY2SEC, mu=self.target_planet.mu_central_body, N=500, color="k", units=AU, axes=ax)
        ax.scatter(0,0,0, color="k", label=self.starting_planet.name)
        ax.legend()

if __name__ == "__main__":
    # Loading the SPICE kernels
    pk.util.load_spice_kernel("sat441.bsp")
    pk.util.load_spice_kernel("de432s.bsp")
    
    # All parameters taken from: https://ssd.jpl.nasa.gov/astro_par.html
    # (and for Titan from: https://solarsystem.nasa.gov/moons/saturn-moons/titan/by-the-numbers/)
    MU_SATURN = 37940584.841800 * 1e9
    MU_TITAN = 8980.14303 * 1e9

    # All parameters taken from: https://solarsystem.nasa.gov/resources/686/solar-system-sizes/
    # (and for Titan from: https://solarsystem.nasa.gov/moons/saturn-moons/titan/by-the-numbers/)
    R_SATURN = 58232 * 1e3
    R_TITAN = 2574.7 * 1e3

    # Spice has arguments: target, observer, ref_frame, abberations, mu_central_body, mu_self, radius, safe_radius
    saturn = pk.planet.spice('SATURN BARYCENTER', 'EARTH', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_SATURN,
                             R_SATURN, R_SATURN)
    saturn.safe_radius = 1.5
    saturn.name = "SATURN"

    #Defining Titan relative to Saturn instead
    titan = pk.planet.spice('TITAN', 'SATURN BARYCENTER', 'ECLIPJ2000', 'NONE', MU_SATURN, MU_TITAN,
                            R_TITAN, R_TITAN)
    titan.name = "TITAN"
    
    start_time = pk.epoch_from_string("2021-DEC-28 11:58:50.816")
    r_target = titan.radius * 2
    e_target = 0.1
    tof = [1, 400]
    r_start_max = 30
    
    udp = PlanetToSatellite(start_time, r_target, e_target, saturn, titan, tof, r_start_max, initial_insertion=True, 
                          v_inf=[-4361.450660605015, -3656.110883139835, 113.24655422161563], max_revs=5)
    prob = pg.problem(udp)
    
    alg_glob = pg.algorithm(pg.mbh(algo=pg.algorithm(pg.gaco(gen=500)),stop=3,perturb=1))
    alg_loc = pg.nlopt('cobyla')
    alg_loc = pg.algorithm(alg_loc)
    
    pop_num = 200
    
    pop = pg.population(prob=udp,size=pop_num)    
    
    print('Global opt')
    pop = alg_glob.evolve(pop)
    
    print('Starting local optimizer')
    pop = alg_loc.evolve(pop)

    champion = pop.champion_x
    
    udp.pretty(champion)
    udp.plot(champion)
    plt.show()
