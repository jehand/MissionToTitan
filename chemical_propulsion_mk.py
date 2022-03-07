import pykep as pk
from pykep.trajopt import mga_1dsm
from pykep.planet import jpl_lp
from pykep import epoch_from_string
import pygmo as pg
from pygmo import *

from rockets import launchers
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from math import log, acos, cos, sin, asin, exp, sqrt


class TitanChemicalUDP(mga_1dsm):
    """
    This class represents a rendezvous mission to Titan modelled as an MGA-1DSM transfer. A launcher model
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

    def __init__(self, sequence, constrained=True):
        """
        The Titan problem of the trajectory gym consists in 48 different instances varying in fly-by sequence and
        the presence of a time constraint.
        Args:
            - sequence (``array``): The sequence defines the fly-by sequence as pykep planet objects.
            - constrained (``bool``): Activates the constraint on the time of flight
              (fitness will thus return two numbers, the objective function and the inequality constraint violation).
        """

        super().__init__(
            seq=sequence,
            t0=[pk.epoch_from_string("2022-JAN-01 00:00:00.000"), pk.epoch_from_string("2030-JAN-01 00:00:00.000")],
            tof=7305.0,
            vinf=[1, 10],
            add_vinf_dep=False,
            add_vinf_arr=True,
            tof_encoding='eta',
            multi_objective=False,
            orbit_insertion=True,
            e_target=.0288,
            rp_target=1186680000,
            eta_lb=0.01,
            eta_ub=0.99,
            rp_ub=25
        )

        self.sequence = sequence
        self.constrained = constrained

    def fitness(self, x):
        T, Vinfx, Vinfy, Vinfz = self._decode_times_and_vinf(x)
        # We transform it (only the needed component) to an equatorial system rotating along x
        # (this is an approximation, assuming vernal equinox is roughly x and the ecliptic plane is roughly xy)
        earth_axis_inclination = 0.409072975
        # This is different from the GTOP tanmEM problem, I think it was bugged there as the rotation was in the wrong direction.
        Vinfz = - Vinfy * sin(earth_axis_inclination) + Vinfz * cos(earth_axis_inclination)
        # And we find the vinf declination (in degrees)
        sindelta = Vinfz / x[3]
        declination = asin(sindelta) / np.pi * 180.
        # We now have the initial mass of the spacecraft
        m_initial = launchers().ariane6(x[3] / 1000., declination) # Need to change the launcher model being used parametrically
        # And we can evaluate the final mass via Tsiolkowsky
        Isp = 312.
        g0 = 9.80665
        DV = super().fitness(x)[0]
        DV = DV + 165.  # losses for 3 swgbys + insertion
        m_final = m_initial * exp(-DV / (Isp * g0))
        # Numerical guard for the exponential
        if m_final == 0:
            m_final = 1e-320
        if self.constrained:
            retval = [-log(m_final), 1500 - m_final]
        else:
            retval = [-log(m_final)]
        return retval

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
        return pg.estimate_gradient_h((lambda x: self.fitness(x), x), dx = 1e-12)

    def pretty(self, x):
        """
        prob.plot(x)
        - x: encoded trajectory
        Prints human readable information on the trajectory represented by the decision vector x
        Example::
          print(prob.pretty(x))
        """
        super().pretty(x)
        T, Vinfx, Vinfy, Vinfz = self._decode_times_and_vinf(x)
        # We transform it (only the needed component) to an equatorial system rotating along x
        # (this is an approximation, assuming vernal equinox is roughly x and the ecliptic plane is roughly xy)
        earth_axis_inclination = 0.409072975
        Vinfz = - Vinfy * sin(earth_axis_inclination) + Vinfz * cos(earth_axis_inclination)
        # And we find the vinf declination (in degrees)
        sindelta = Vinfz / x[3]
        declination = asin(sindelta) / np.pi * 180.
        m_initial = launchers().ariane6(x[3] / 1000., declination)
        # And we can evaluate the final mass via Tsiolkowsky
        Isp = 312.
        g0 = 9.80665
        DV = super().fitness(x)[0]
        DV = DV + 165.  # losses for 3 swgbys + insertion
        m_final = m_initial * exp(-DV / (Isp * g0))
        print("\nInitial mass:", m_initial)
        print("Final mass:", m_final)
        print("Declination:", declination)

    def __repr__(self):
        return "AEON (Trajectory Optimisation for a Rendezvous with Titan)"

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
                              R_JUPITER, R_JUPITER*1.1)
   

    saturn = pk.planet.spice('SATURN BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_SATURN,
                             R_SATURN, R_SATURN*1.1)
   


    # Defining the sequence and the problem
    planetary_sequence = [earth,mars,jupiter]
    udp = TitanChemicalUDP(sequence=planetary_sequence, constrained=True)
    print(udp)
    # We solve it!!
    tada = pg.algorithm(compass_search())
    # tada.ftol_rel = 1e-12
    # tada.ftol_abs = 1e-10
    uda = pg.algorithms.mbh(algo=tada)

    archi = pg.archipelago(algo=uda, prob=udp, n=8, pop_size=20)

    archi.evolve(20)
    archi.wait()
    sols = archi.get_champions_f()
    sols2 = [item[0] for item in sols]
    idx = sols2.index(min(sols2))
    champion = archi.get_champions_x()[idx]
    udp.pretty(champion)
    
    # print(pg.problem(udp).feasibility_x(champion))
    # print(champion)
    
    
    
    mpl.rcParams['legend.fontsize'] = 6
    
    fig = plt.figure()
    axis = fig.add_subplot(projection='3d')
    udp.plot(champion, ax=axis)
    axis.legend(fontsize=6)
    plt.show()
    
    
    """
    Problem definition: 
    1) MGA_1DSM to Saturn
    2) Once under Saturn's influence, switch to Saturns point of reference and do a PL2PL to Titan
    3) Once under Titan's influence, burn into the orbit we are concerned with
    Constraints:
    1) Vinf at Titan
    2) Orbit at Titan
    3) Payload mass
    4) Vinf at Saturn maybe?
    6) Launcher capability
    
    Minimizing:
    1) Mass difference (fuel consumption: (m0-mf) / m0)
    2) Time of flight
    
    Notes:
    1) Have a feasibility while loop after to repeat until you get feasible results (try catch)
    2) Set Saturn's orbit insertion to be between its rings
    
    MGA_1DSM Decision Vector:
                         0     1  2   3     4     5      6      7     8    9   ....    -1
      direct encoding: [t0] + [u, v, Vinf, eta1, T1] + [beta, rp/rV, eta2, T2] + ... 
      alpha encoding:  [t0] + [u, v, Vinf, eta1, a1] + [beta, rp/rV, eta2, a2] + ... + [T]
      eta encoding:    [t0] + [u, v, Vinf, eta1, n1] + [beta, rp/rV, eta2, n2] + ...
      
      where t0 is a mjd2000, Vinf is in km/s, T in days, beta in radians and the rest non dimensional.
    """