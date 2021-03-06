import pykep as pk
from pykep.trajopt import mga_1dsm
from pykep.core import epoch, DAY2SEC, MU_SUN, lambert_problem, propagate_lagrangian, fb_prop, AU, epoch
from pykep.trajopt._lambert import lambert_problem_multirev
import pygmo as pg
import numpy as np
from math import log, acos, cos, sin, asin, exp, sqrt
from typing import Any, Dict, List, Tuple

try:
    from rockets import launchers
    from algorithms import Algorithms
except:
    import sys
    sys.path.append(sys.path[0]+"/udps")
    from rockets import launchers
    from algorithms import Algorithms

# Avoiding scipy dependency
def norm(x):
    return sqrt(sum([it * it for it in x]))

class TitanChemical1DSMUDP(mga_1dsm):
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
            t0=[pk.epoch_from_string("2027-JAN-01 00:00:00.000"), pk.epoch_from_string("2030-JAN-01 00:00:00.000")],
            tof=[[20, 1500] for _ in range(len(sequence)-1)],
            vinf=[1, 7],
            add_vinf_dep=False,
            add_vinf_arr=True,
            tof_encoding='direct',
            multi_objective=False,
            orbit_insertion=False,
            e_target=0.98531407996358,
            rp_target=80330000,
            eta_lb=0.01,
            eta_ub=0.99,
            rp_ub=10
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
        Isp = 312.
        g0 = 9.80665
        DV = super().fitness(x)[0]
        DV = DV + 165.  # losses for 3 swgbys + insertion
        m_final = m_initial * exp(-DV / (Isp * g0))
        # Numerical guard for the exponential
        if m_final == 0:
            m_final = 1e-320
        if self.constrained:
            retval = [-log(m_final), sum(T) - 3652.5]
        else:
            retval = [DV, ]#[DV + sqrt(Vinfx**2 + Vinfy**2 + Vinfz**2), ]
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
        return pg.estimate_gradient((lambda x: self.fitness(x), x), dx = 1e-8)

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
        print("\nInitial mass: {0:.4g}".format(m_initial[0]), "kg")
        print("Final mass: {0:.4g}".format(m_final[0]), "kg")
        print("Declination: {0:.4g}".format(declination), "degrees")
        print("Interplanetary DV: {0:.4g}".format(DV/1000), "km/s")

    # Plot of the trajectory
    def plot(self, x, ax = None):
        """
        ax = prob.plot(x, ax=None)

        - x: encoded trajectory
        - ax: matplotlib axis where to plot. If None figure and axis will be created
        - [out] ax: matplotlib axis where to plot

        Plots the trajectory represented by a decision vector x on the 3d axis ax

        Example::

          ax = prob.plot(x)
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from pykep.orbit_plots import plot_planet, plot_lambert, plot_kepler

        if ax is None:
            mpl.rcParams['legend.fontsize'] = 10
            fig = plt.figure()
            axis = fig.gca(projection='3d')
        else:
            axis = ax

        axis.scatter(0, 0, 0, color='y')

        # 1 -  we 'decode' the chromosome recording the various times of flight
        # (days) in the list T and the cartesian components of vinf
        T, Vinfx, Vinfy, Vinfz = self._decode_times_and_vinf(x)

        # 2 - We compute the epochs and ephemerides of the planetary encounters
        t_P = list([None] * (self.n_legs + 1))
        r_P = list([None] * (self.n_legs + 1))
        v_P = list([None] * (self.n_legs + 1))
        DV = list([None] * (self.n_legs + 1))

        for i, planet in enumerate(self._seq):
            t_P[i] = epoch(x[0] + sum(T[0:i]))
            r_P[i], v_P[i] = planet.eph(t_P[i])
            plot_planet(planet, t0=t_P[i], color=(
                1.0/(i+1), 0.6, 0.8), legend=True, units=AU, axes=axis, N=150)

        # 3 - We start with the first leg
        v0 = [a + b for a, b in zip(v_P[0], [Vinfx, Vinfy, Vinfz])]
        r, v = propagate_lagrangian(
            r_P[0], v0, x[4] * T[0] * DAY2SEC, self.common_mu)

        plot_kepler(r_P[0], v0, x[4] * T[0] * DAY2SEC, self.common_mu,
                    N=100, color='b', units=AU, axes=axis)

        # Lambert arc to reach seq[1]
        dt = (1 - x[4]) * T[0] * DAY2SEC
        
        l = lambert_problem_multirev(v, lambert_problem(
            r, r_P[1], dt, self.common_mu, cw=False, max_revs=self.max_revs))
        
        plot_lambert(l, sol=0, color='r', units=AU, axes=axis)
        v_end_l = l.get_v2()[0]
        v_beg_l = l.get_v1()[0]

        # 4 - And we proceed with each successive leg
        for i in range(1, self.n_legs):
            # Fly-by
            v_out = fb_prop(v_end_l, v_P[i], x[
                            7 + (i - 1) * 4] * self._seq[i].radius, x[6 + (i - 1) * 4], self._seq[i].mu_self)
            # s/c propagation before the DSM
            r, v = propagate_lagrangian(
                r_P[i], v_out, x[8 + (i - 1) * 4] * T[i] * DAY2SEC, self.common_mu)
            plot_kepler(r_P[i], v_out, x[8 + (i - 1) * 4] * T[i] * DAY2SEC,
                        self.common_mu, N=100, color='b', units=AU, axes=axis)
            # Lambert arc to reach Earth during (1-nu2)*T2 (second segment)
            dt = (1 - x[8 + (i - 1) * 4]) * T[i] * DAY2SEC

            l = lambert_problem_multirev(v, lambert_problem(r, r_P[i + 1], dt,
                self.common_mu, cw=False, max_revs=self.max_revs))

            plot_lambert(l, sol=0, color='r', legend=False,
                         units=AU, N=1000, axes=axis)
            
        return axis

    def __repr__(self):
        return "AEON (Trajectory Optimisation for a Rendezvous with Titan)"

if __name__ == "__main__":
    pk.util.load_spice_kernel("sat441.bsp")
    pk.util.load_spice_kernel("de432s.bsp")
    
    # All parameters taken from: https://ssd.jpl.nasa.gov/astro_par.html
    # (and for Titan from: https://solarsystem.nasa.gov/moons/saturn-moons/titan/by-the-numbers/)
    MU_VENUS = 324858.592000 * 1e9
    MU_MARS = 42828.375816 * 1e9
    MU_JUPITER = 126712764.100000 * 1e9
    MU_SATURN = 37940584.841800 * 1e9
    MU_TITAN = 8980.14303 * 1e9

    # All parameters taken from: https://solarsystem.nasa.gov/resources/686/solar-system-sizes/
    # (and for Titan from: https://solarsystem.nasa.gov/moons/saturn-moons/titan/by-the-numbers/)
    R_VENUS = 6052 * 1e3
    R_MARS = 3390 * 1e3
    R_JUPITER = 69911 * 1e3
    R_SATURN = 58232 * 1e3
    R_TITAN = 2574.7 * 1e3

    # Spice has arguments: target, observer, ref_frame, abberations, mu_central_body, mu_self, radius, safe_radius
    earth = pk.planet.spice('EARTH BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, pk.MU_EARTH,
                            pk.EARTH_RADIUS, pk.EARTH_RADIUS * 1.05)

    venus = pk.planet.spice('VENUS BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_VENUS,
                            R_VENUS, R_VENUS)
    venus.safe_radius = 1.05

    mars = pk.planet.spice('MARS BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_MARS,
                           R_MARS, R_MARS)
    mars.safe_radius = 1.05

    jupiter = pk.planet.spice('JUPITER BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_JUPITER,
                              R_JUPITER, R_JUPITER)
    jupiter.safe_radius = 1.7

    saturn = pk.planet.spice('SATURN BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_SATURN,
                             R_SATURN, R_SATURN)
    saturn.safe_radius = 1.5

    titan = pk.planet.spice('TITAN', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_TITAN,
                            R_TITAN, R_TITAN)

    # Defining the sequence and the problem
    planetary_sequence = [earth, venus, mars, jupiter, saturn]
    udp = TitanChemicalUDP(sequence=planetary_sequence, constrained=False)


    # We solve it!!
    sol = Algorithms(problem=udp)
    #uda = sol.self_adaptive_differential_algorithm()
    uda = sol.augmented_lagrangian(local_algo="slsqp")
    champion = sol.archipelago(uda, islands=8, island_population=20)

    udp.pretty(champion)
    #print(pg.problem(udp).feasibility_x(champion))
    #print(champion)
    
    # mpl.rcParams['legend.fontsize'] = 6
    
    # fig = plt.figure()
    # axis = fig.add_subplot(projection='3d')
    # udp.plot(champion, ax=axis)
    # axis.legend(fontsize=6)
    # plt.show()