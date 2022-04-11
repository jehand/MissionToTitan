import pykep as pk
from pykep.trajopt import mga_1dsm
from pykep.core import epoch, DAY2SEC, MU_SUN, lambert_problem, propagate_lagrangian, fb_prop, AU, epoch
from pykep.trajopt._lambert import lambert_problem_multirev
import pygmo as pg
from pygmo import *
import numpy as np
from math import log, acos, cos, sin, asin, exp, sqrt
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt

try:
    from rockets import launchers
    from algorithms import Algorithms
except:
    import sys
    sys.path.append(sys.path[0]+"/udps")
    from rockets import launchers
    from algorithms import Algorithms

def norm(x):
    return sqrt(sum([it * it for it in x]))

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

    def __init__(self, sequence, constrained=False):
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
            t0=[pk.epoch_from_string("1997-JAN-01 00:00:00.000"), pk.epoch_from_string("1997-DEC-31 00:00:00.000")],
            tof=2500,
            vinf=[4, 5],
            add_vinf_dep=False,
            add_vinf_arr=True,
            tof_encoding='eta',
            multi_objective=False,
            orbit_insertion=True,
            e_target=.9823,
            rp_target=78232 * 1e3,
            rp_ub=300,
            max_revs= 3,
            eta_ub = .99,
            eta_lb = .01
        )

        self.sequence = sequence
        self.constrained = constrained

    def _compute_dvs(self, x: List[float]) -> Tuple[
        List[float], # DVs
        List[Any], # Lambert legs
        List[float], # T
        List[Tuple[List[float], List[float]]], # ballistic legs
        List[float], # epochs of ballistic legs
    ]:
        # 1 -  we 'decode' the chromosome recording the various times of flight
        # (days) in the list T and the cartesian components of vinf
        T, Vinfx, Vinfy, Vinfz = self._decode_times_and_vinf(x)

        # 2 - We compute the epochs and ephemerides of the planetary encounters
        t_P = list([None] * (self.n_legs + 1))
        r_P = list([None] * (self.n_legs + 1))
        v_P = list([None] * (self.n_legs + 1))
        DV = list([0.0] * (self.n_legs + 1))
        for i in range(len(self._seq)):
            t_P[i] = epoch(x[0] + sum(T[0:i]))
            r_P[i], v_P[i] = self._seq[i].eph(t_P[i])
        ballistic_legs: List[Tuple[List[float],List[float]]] = []
        ballistic_ep: List[float] = []
        lamberts = []

        # 3 - We start with the first leg
        v0 = [a + b for a, b in zip(v_P[0], [Vinfx, Vinfy, Vinfz])]
        ballistic_legs.append((r_P[0], v0))
        ballistic_ep.append(t_P[0].mjd2000)
        r, v = propagate_lagrangian(
            r_P[0], v0, x[4] * T[0] * DAY2SEC, self.common_mu)

        # Lambert arc to reach seq[1]
        dt = (1 - x[4]) * T[0] * DAY2SEC
        l = lambert_problem_multirev(v, lambert_problem(
                    r, r_P[1], dt, self.common_mu, cw=False, max_revs=self.max_revs))
        v_end_l = l.get_v2()[0]
        v_beg_l = l.get_v1()[0]
        lamberts.append(l)

        ballistic_legs.append((r, v_beg_l))
        ballistic_ep.append(t_P[0].mjd2000 + x[4] * T[0])

        # First DSM occuring at time nu1*T1
        DV[0] = norm([a - b for a, b in zip(v_beg_l, v)])

        # 4 - And we proceed with each successive leg
        for i in range(1, self.n_legs):
            # Fly-by
            v_out = fb_prop(v_end_l, v_P[i], x[
                            7 + (i - 1) * 4] * self._seq[i].radius, x[6 + (i - 1) * 4], self._seq[i].mu_self)
            ballistic_legs.append((r_P[i], v_out))
            ballistic_ep.append(t_P[i].mjd2000)
            # s/c propagation before the DSM
            r, v = propagate_lagrangian(
                r_P[i], v_out, x[8 + (i - 1) * 4] * T[i] * DAY2SEC, self.common_mu)
            # Lambert arc to reach Earth during (1-nu2)*T2 (second segment)
            dt = (1 - x[8 + (i - 1) * 4]) * T[i] * DAY2SEC
            l = lambert_problem_multirev(v, lambert_problem(r, r_P[i + 1], dt,
                                  self.common_mu, cw=False, max_revs=self.max_revs))
            v_end_l = l.get_v2()[0]
            v_beg_l = l.get_v1()[0]
            lamberts.append(l)
            # DSM occuring at time nu2*T2
            DV[i] = norm([a - b for a, b in zip(v_beg_l, v)])

            ballistic_legs.append((r, v_beg_l))
            ballistic_ep.append(t_P[i].mjd2000 + x[8 + (i - 1) * 4] * T[i])

        # Last Delta-v
        if self._add_vinf_arr:
            DV[-1] = norm([a - b for a, b in zip(v_end_l, v_P[-1])])
            if self._orbit_insertion:
                # In this case we compute the insertion DV as a single pericenter
                # burn
                DVper = np.sqrt(DV[-1] * DV[-1] + 2 *
                                self._seq[-1].mu_self / self._rp_target)
                DVper2 = np.sqrt(2 * self._seq[-1].mu_self / self._rp_target -
                                self._seq[-1].mu_self / self._rp_target * (1. - self._e_target))
                DV[-1] = np.abs(DVper - DVper2)

        if self._add_vinf_dep:
            DV[0] += x[3]

        return (DV, lamberts, T, ballistic_legs, ballistic_ep)

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
        Isp = 324.
        g0 = 9.80665
        DV = super().fitness(x)[0]
        DV = DV + 165.  # losses for 3 swgbys + insertion
        m_final = m_initial * exp(-DV / (Isp * g0))
        # Numerical guard for the exponential
        if m_final == 0:
            m_final = 1e-320
        if self.constrained:
            retval = [DV, 3000-m_final]
        else:
            retval = [DV]
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
        return pg.estimate_gradient(lambda x: self.fitness(x), x)

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
        Isp = 324.
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
                              R_JUPITER, R_JUPITER*1.05)
   

    saturn = pk.planet.spice('SATURN BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_SATURN,
                             R_SATURN, R_SATURN*1.05)
   


    # Defining the sequence and the problem
    planetary_sequence = [earth,venus,venus,earth,jupiter,saturn]
    udp = TitanChemicalUDP(sequence=planetary_sequence, constrained=False)
    print(udp)
    # We solve it!!
        
    
    alg_glob = pg.algorithm(pg.mbh(algo=pg.algorithm(pg.gaco(gen=2000)),stop=7,perturb=1))
    alg_loc = pg.nlopt('cobyla')
    alg_loc = pg.algorithm(alg_loc)
    
    verb = 500
    pop_num = 1000
    
    pop = pg.population(prob=udp,size=pop_num)    
    
    alg_glob.set_verbosity(verb)
    alg_loc.set_verbosity(verb)
    
    print('Global opt')
    pop = alg_glob.evolve(pop)
    
    print('starting local optimizer')
    pop = alg_loc.evolve(pop)
    


    champion = pop.champion_x
    udp.pretty(champion)

    
    # print(pg.problem(udp).feasibility_x(champion))
    # print(champion)
    
    
    
    # mpl.rcParams['legend.fontsize'] = 6
    
    # fig = plt.figure()
    # axis = fig.add_subplot(projection='3d')
    # udp.plot([-7.93317459e+02,  9.46312140e-01,  3.86191253e-01,  3.97920748e+03,
    #   7.24980923e-01,  7.48075354e-02,  4.59262635e+00,  1.58578898e+00,
    #   4.19396583e-01,  1.80123171e-01,  4.32493584e+00,  1.10000622e+00,
    #   1.05229434e-02,  2.85468085e-02, -1.55886225e+00,  1.10002510e+00,
    #   1.07062882e-02,  2.65583612e-01, -1.58693224e+00,  1.85045334e+02,
    #   2.36245211e-02,  8.43569703e-01], ax=axis)
    # axis.legend(fontsize=6)
    # plt.show()
    
    
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