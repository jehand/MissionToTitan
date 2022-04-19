import pykep as pk
from pykep.trajopt import mga
from pykep.core import epoch, DAY2SEC, MU_SUN, lambert_problem, propagate_lagrangian, fb_vel, AU, epoch
from pykep.trajopt._lambert import lambert_problem_multirev
import pygmo as pg
from pygmo import *
import numpy as np
from math import log, acos, cos, sin, asin, exp, sqrt
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
from bisect import bisect_left

try:
    from rockets import launchers
    #from algorithms import Algorithms
except:
    import sys
    sys.path.append(sys.path[0]+"/udps")
    from rockets import launchers
    #from algorithms import Algorithms

class TitanChemicalMGAUDP(mga):
    """
    This class represents a rendezvous mission to Titan modelled as an MGA transfer. A launcher model
    is also used, so that the final mass delivered to Saturn, and the Delta V are the main objectives of this
    optimization problem.
    .. note::
       The class is currently set up to only optimize for optimal mass. Things that need to be improved:
       1) Make more parameters parametric, such as rp_target, etc.
       2) Switch to optimizing for Delta-V
       3) Include a spacecraft sizing part for the optimization
       4) Clarify whether the fitness function is correct for our mission
       5) Remove the time constraint
    """

    def __init__(self, sequence, departure_range=[pk.epoch_from_string("1997-JAN-01 00:00:00.000"), pk.epoch_from_string("1997-DEC-31 00:00:00.000")], constrained=False):
        """
        The Titan problem of the trajectory gym consists in 48 different instances varying in fly-by sequence and
        the presence of a time constraint.
        Args:
            - sequence (``array``): The sequence defines the fly-by sequence as pykep planet objects.
            - departure_range (``array``): The range of departure dates as [lower bound, upper bound]
            - constrained (``bool``): Activates the constraint on the time of flight
              (fitness will thus return two numbers, the objective function and the inequality constraint violation).
        """

        super().__init__(
            seq=sequence,
            t0=departure_range,
            tof=3500,
            vinf=4,
            tof_encoding='eta',
            multi_objective=False,
            orbit_insertion=True,
            e_target=.99,
            rp_target=101906881,
            max_revs=3,
        )

        self.sequence = sequence
        self.constrained = constrained

    def _compute_dvs(self, x: List[float]) -> Tuple[
        float, # DVlaunch
        List[float], # DVs
        float, # DVarrival,
        List[Any], # Lambert legs
        float, #DVlaunch_tot
        List[float], # T
        List[Tuple[List[float], List[float]]], # ballistic legs
        List[float], # epochs of ballistic legs
    ]:
        # 1 -  we 'decode' the times of flights and compute epochs (mjd2000)
        T: List[float] = self._decode_tofs(x)  # [T1, T2 ...]
        ep = np.insert(T, 0, x[0])  # [t0, T1, T2 ...]
        ep = np.cumsum(ep)  # [t0, t1, t2, ...]
        # 2 - we compute the ephemerides
        r = [0] * len(self.seq)
        v = [0] * len(self.seq)
        for i in range(len(self.seq)):
            r[i], v[i] = self.seq[i].eph(float(ep[i]))

        l = list()
        ballistic_legs: List[Tuple[List[float],List[float]]] = []
        ballistic_ep: List[float] = []

        # 3 - we solve the lambert problems
        vi = v[0]
        for i in range(self._n_legs):
            lp = lambert_problem_multirev(
                vi, lambert_problem(
                    r[i], r[i + 1], T[i] * DAY2SEC, self._common_mu, False, self.max_revs))
            l.append(lp)
            vi = lp.get_v2()[0]
            ballistic_legs.append((r[i], lp.get_v1()[0]))
            ballistic_ep.append(ep[i])
        # 4 - we compute the various dVs needed at fly-bys to match incoming
        # and outcoming
        DVfb = list()
        for i in range(len(l) - 1):
            vin = [a - b for a, b in zip(l[i].get_v2()[0], v[i + 1])]
            vout = [a - b for a, b in zip(l[i + 1].get_v1()[0], v[i + 1])]
            DVfb.append(fb_vel(vin, vout, self.seq[i + 1]))
        # 5 - we add the departure and arrival dVs
        DVlaunch_tot = np.linalg.norm(
            [a - b for a, b in zip(v[0], l[0].get_v1()[0])])
        DVlaunch = max(0, DVlaunch_tot - self.vinf)
        DVarrival = np.linalg.norm(
            [a - b for a, b in zip(v[-1], l[-1].get_v2()[0])])
        if self.orbit_insertion:
            # In this case we compute the insertion DV as a single pericenter
            # burn
            DVper = np.sqrt(DVarrival * DVarrival + 2 *
                            self.seq[-1].mu_self / self.rp_target)
            DVper2 = np.sqrt(2 * self.seq[-1].mu_self / self.rp_target -
                             self.seq[-1].mu_self / self.rp_target * (1. - self.e_target))
            DVarrival = np.abs(DVper - DVper2)
        return (DVlaunch, DVfb, DVarrival, l, DVlaunch_tot, T, ballistic_legs, ballistic_ep)

    def fitness(self, x):
        DVlaunch, DVfb, DVarrival, _, _, _, _, _ = self._compute_dvs(x)
        if self.tof_encoding == 'direct':
            T = sum(x[1:])
        elif self.tof_encoding == 'alpha':
            T = x[1]
        elif self.tof_encoding == 'eta':
            T = sum(self.eta2direct(x)[1:])
        if self.multi_objective:
            return [DVlaunch + np.sum(DVfb) + DVarrival, T]
        else:
            return [DVlaunch + np.sum(DVfb) + DVarrival]

    def get_nic(self):
        return int(self.constrained)

    def get_name(self):
        return "AEON sequence: " + str(self.sequence)

    def get_extra_info(self):
        retval = "\t Sequence: " + \
                 [pl.name for pl in self._seq].__repr__() + "\n\t Constrained: " + \
                 str(self.constrained)
        return retval

    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)

    def pretty(self, x):
        """
        prob.plot(x)
        - x: encoded trajectory
        Prints human readable information on the trajectory represented by the decision vector x
        Example::
          print(prob.pretty(x))
        """
        T = self._decode_tofs(x)
        ep = np.insert(T, 0, x[0])  # [t0, T1, T2 ...]
        ep = np.cumsum(ep)  # [t0, t1, t2, ...]
        DVlaunch, DVfb, DVarrival, l, DVlaunch_tot, _, _, _ = self._compute_dvs(x)
        print("Multiple Gravity Assist (MGA) problem: ")
        print("Planet sequence: ", [pl.name for pl in self.seq])

        print("Departure: ", self.seq[0].name)
        print("\tDate: ", pk.epoch(ep[0]))
        print("\tEpoch: ", ep[0], " [mjd2000]")
        print("\tSpacecraft velocity: ", l[0].get_v1()[0], "[m/s]")
        print("\tHyperbolic velocity: ", DVlaunch_tot, "[m/s]")
        print("\tInitial DV: ", DVlaunch, "[m/s]")

        for pl, e, dv in zip(self.seq[1:-1], ep[1:-1], DVfb):
            print("Fly-by: ", pl.name)
            print("\tDate: ", pk.epoch(e))
            print("\tEpoch: ", e, " [mjd2000]")
            print("\tDV: ", dv, "[m/s]")

        print("Arrival: ", self.seq[-1].name)
        print("\tDate: ", pk.epoch(ep[-1]))
        print("\tEpoch: ", ep[-1], " [mjd2000]")
        print("\tSpacecraft velocity: ", l[-1].get_v2()[0], "[m/s]")
        print("\tArrival DV: ", DVarrival, "[m/s]")

        print("Time of flights: ", T, "[days]")
        
        print("Total DV (excluding arrival) = {:.3f} km/s".format((DVlaunch + np.sum(DVfb))/1000))
        print("Time of departure =", pk.epoch(x[0]))
        print("Arrival date =", pk.epoch(x[0] + sum(T)))
        print("Total Time of Flight =", sum(T)/365, "[years] or", sum(T), "[days]")

    def plot(self, x, ax=None, units=AU, N=60):
        """plot(self, x, ax=None, units=pk.AU, N=60)
        Plots the spacecraft trajectory.
        Args:
            - x (``tuple``, ``list``, ``numpy.ndarray``): Decision chromosome.
            - ax (``matplotlib.axes._subplots.Axes3DSubplot``): 3D axes to use for the plot
            - units (``float``, ``int``): Length unit by which to normalise data.
            - N (``float``): Number of points to plot per leg
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        from pykep.orbit_plots import plot_planet, plot_lambert

        # Creating the axes if necessary
        if ax is None:
            mpl.rcParams['legend.fontsize'] = 10
            fig = plt.figure()
            ax = fig.gca(projection='3d')

        _, _, _, l, _, T, _, _ = self._compute_dvs(x)
        ep = np.insert(T, 0, x[0])  # [t0, T1, T2 ...]
        ep = np.cumsum(ep)  # [t0, t1, t2, ...]
        
        for pl, e in zip(self.seq, ep):
            plot_planet(pl, epoch(e), units=units, legend=True,
                        color=(0.7, 0.7, 1), axes=ax)
        for lamb in l:
            plot_lambert(lamb, N=N, sol=0, units=units, color='r',
                         legend=False, axes=ax, alpha=0.8)
        return ax

    def get_eph_function(self, x: List[float]):
        """
        For a chromosome x, returns a function object eph to compute the ephemerides of the spacecraft
        Args:
            - x (``list``, ``tuple``, ``numpy.ndarray``): Decision chromosome, e.g. (``pygmo.population.champion_x``).
        Example:
          eph = prob.get_eph_function(population.champion_x)
          pos, vel = eph(pykep.epoch(7000))
        """
        if len(x) != len(self.get_bounds()[0]):
            raise ValueError(
                "Expected chromosome of length "
                + str(len(self.get_bounds()[0]))
                + " but got length "
                + str(len(x))
            )

        _, _, _, _, _, _, b_legs, b_ep = self._compute_dvs(x)
        
        def eph(
            t: float
        ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:

            if t < b_ep[0]:
                raise ValueError(
                    "Given epoch " + str(t) + " is before launch date " + str(b_ep[0])
                )

            if t == b_ep[0]:
                # exactly at launch
                return self.seq[0].eph(t)

            i = bisect_left(b_ep, t)  # ballistic leg i goes from planet i to planet i+1

            assert i >= 1 and i <= len(b_ep)
            if i < len(b_ep):
                assert t <= b_ep[i]

            # get start of ballistic leg
            r_b, v_b = b_legs[i - 1]

            elapsed_seconds = (t - b_ep[i - 1]) * DAY2SEC
            assert elapsed_seconds >= 0

            # propagate the lagrangian
            r, v = propagate_lagrangian(r_b, v_b, elapsed_seconds, self.seq[0].mu_central_body)

            return r, v
        
        return eph

    def __repr__(self):
        return "AEON MGA (Trajectory Optimisation for a Rendezvous with Titan)"

if __name__ == "__main__":

    pk.util.load_spice_kernel('de430.bsp')
    
    # All parameters taken from: https://ssd.jpl.nasa.gov/astro_par.html
    # (and for Titan from: https://solarsystem.nasa.gov/moons/saturn-moons/titan/by-the-numbers/)
    MU_VENUS = 324858.592000 * 1e9
    MU_MARS = 42828.375816 * 1e9
    MU_JUPITER = 126712764.100000 * 1e9
    MU_SATURN = 37940584.841800 * 1e9

    # All parameters taken from: https://solarsystem.nasa.gov/resources/686/solar-system-sizes/
    # (and for Titan from: https://solarsystem.nasa.gov/moons/saturn-moons/titan/by-the-numbers/)
    R_VENUS = 6052 * 1e3
    R_MARS = 3390 * 1e3
    R_JUPITER = 69911 * 1e3
    R_SATURN = 58232 * 1e3

    # Spice has arguments: target, observer, ref_frame, abberations, mu_central_body, mu_self, radius, safe_radius
    earth = pk.planet.spice('EARTH BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, pk.MU_EARTH,
                            pk.EARTH_RADIUS, pk.EARTH_RADIUS * 1.1)

    venus = pk.planet.spice('VENUS BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_VENUS,
                            R_VENUS, R_VENUS*1.1)
    

    mars = pk.planet.spice('MARS BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_MARS,
                           R_MARS, R_MARS*1.1)
   

    jupiter = pk.planet.spice('JUPITER BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_JUPITER,
                              R_JUPITER, R_JUPITER*1.05)
   

    saturn = pk.planet.spice('SATURN BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_SATURN,
                             R_SATURN, R_SATURN*1.05)
   
    # Defining the sequence and the problem
    planetary_sequence = [earth,venus,venus,earth,jupiter,saturn]
    udp = TitanChemicalMGAUDP(sequence=planetary_sequence, constrained=False)
    print(udp)
    # We solve it!!

    alg_glob = pg.algorithm(pg.mbh(algo=pg.algorithm(pg.gaco(gen=20)),stop=7,perturb=1))
    alg_loc = pg.nlopt('cobyla')
    alg_loc = pg.algorithm(alg_loc)
    
    pop_num = 100
    
    pop = pg.population(prob=udp,size=pop_num)    
    
    print('Global opt')
    pop = alg_glob.evolve(pop)
    
    print('starting local optimizer')
    pop = alg_loc.evolve(pop)
    
    champion = pop.champion_x
    udp.pretty(champion)
