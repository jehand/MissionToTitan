import pykep as pk
from pykep.trajopt import mga_1dsm, launchers
from pykep.trajopt._lambert import lambert_problem_multirev
from pykep import epoch_from_string
from pykep.core import epoch, DAY2SEC, MU_SUN, lambert_problem, propagate_lagrangian, fb_prop, AU, epoch
import pygmo as pg

from typing import Any, Dict, List, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
from algorithms import Algorithms
from rockets import launchers
import numpy as np
from math import log, acos, cos, sin, asin, exp, sqrt

# Avoiding scipy dependency
def norm(x):
    return sqrt(sum([it * it for it in x]))

class TitanChemicalUDP(mga_1dsm):
    """
    This class represents a rendezvous mission to Titan modelled as an MGA-1DSM transfer. A launcher model
    is also used, so that the final mass delivered to Saturn, and the Delta V are the main objectives of this
    optimization problem.

    .. note::

       The class is currently set up to only optimize for optimal mass. Things that need to be improved:
       1) Switch to minimizing difference between desired mass delivered and final mass possible (and also Delta-V?)
       2) Include a spacecraft sizing part to go around the optimization
       3) Check fitness function
       4) Check whether coordinate frames are being adjusted correctly
    """

    def __init__(self, sequence, launch_window=[pk.epoch_from_string("2021-DEC-28 11:58:50.816"),
                                                pk.epoch_from_string("2027-DEC-28 11:58:50.816")],
                 Isp=312, g0=9.80665, launcher=launchers().ariane6, final_orbit_radius=80330000,
                 final_orbit_eccentricity=0.98531407996358, adapt_last_leg=True, final_mass=10, constrained=False):
        """
        The Titan problem is that of minimizing the difference between initial and final mass (reducing fuel usage).
        Constrained can be set to true to put a constraint on the time of flight.

        Args:
            - sequence (``array (pykep.planet)``): The sequence defines the fly-by sequence as pykep planet objects.
            - launch_window (``array (pk.epoch)``): The range of dates for which the launch can happen between.
            - Isp (``float``): Specific impulse of the spacecraft's propulsion system.
            - g0 (``float``): Gravity at the place of launch.
            - launcher (``rockets.launchers``): User specified rocket interp2d model for mass at a given declination and
                                                V_inf.
            - final_orbit_radius (``float``): The final orbit radius desired at the target planet (in metres).
            - final_orbit_eccentricity (``float``): The final orbit eccentricity desired at the target planet.
            - adapt_last_leg (``bool``): Adapts the last leg to consider the second last planet as the largest
                                         gravitational impact. The last part of the trajectory to the last planet in
                                         sequence will then be considered differently.
            - final_mass (``float``): Sets the minimum final mass that will be used as a constraint (m_f >= final_mass)
            - constrained (``bool``): Activates the constraint on the time of flight
              (fitness will thus return two numbers, the objective function and the inequality constraint violation).
        """

        super().__init__(
            seq=sequence,
            t0=launch_window,
            tof=[[20, 1000] for _ in range(len(sequence)-1)],
            vinf=[2.5, 4.9],
            add_vinf_dep=False,
            add_vinf_arr=True,
            tof_encoding='direct',
            multi_objective=False,
            orbit_insertion=True,
            e_target=final_orbit_eccentricity,
            rp_target=final_orbit_radius,
            eta_lb=0.01,
            eta_ub=0.99,
            rp_ub=10
        )

        if adapt_last_leg: # Changing the sequence to only be
            self.seq = sequence[:-1]

        self.sequence = sequence
        self.Isp = Isp
        self.g0 = g0
        self.adapt_last_leg = adapt_last_leg
        self.final_mass = final_mass
        self.constrained = constrained

        self.rockets = launcher

    def _compute_dvs(self, x: List[float]) -> Tuple[
        List[float], # DVs
        List[Any], # Lambert legs
        List[float], # T (time of each leg)
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
            
            # Converting into Saturn's reference frame (if adapt_last_leg is True, the last leg will be 
            # calculated relative to the second last planet)
            if (i == self.n_legs - 1) and self.adapt_last_leg:
                # Step 1: find the coordinates and velocity relative to Saturn's barycenter
                r_new = [0,0,0] #assuming we are at the barycenter iniitally?
                v_new = [a-b for a,b in zip(v_out,v_P[i])]
                
                # Step 2: repeat the usual steps with the new coordinates and adjust the common mu
                r, v = propagate_lagrangian(
                    r_new, v_new, x[8 + (i - 1) * 4] * T[i] * DAY2SEC, self._seq[i].mu_self)
                # Lambert arc to reach next planet in sequence during (1-nu2)*T2 (second segment)
                dt = (1 - x[8 + (i - 1) * 4]) * T[i] * DAY2SEC
                l = lambert_problem_multirev(v, lambert_problem(r, [a-b for a,b in zip(r_P[i + 1],self._seq[i].eph(t_P[i+1])[0])], dt,
                                    self._seq[i].mu_self, cw=False, max_revs=self.max_revs))
                v_end_l = l.get_v2()[0]
                v_beg_l = l.get_v1()[0]
                lamberts.append(l)
                # DSM occuring at time nu2*T2
                DV[i] = norm([a - b for a, b in zip(v_beg_l, v)])

                # Step 3: adjust coordinates back to the usual frame to carry out the rest of the analysis
                r_Pmid, v_Pmid = self._seq[i].eph(t_P[i] + x[8 + (i - 1) * 4] * T[i] * DAY2SEC)
                r = [a+b for a,b in zip(r,r_Pmid)]
                v_end_l = [a+b for a,b in zip(v_end_l,v_P[i+1])]
                v_beg_l = [a+b for a,b in zip(v_beg_l,v_Pmid)]
                
                ballistic_legs.append((r, v_beg_l))
                ballistic_ep.append(t_P[i].mjd2000 + x[8 + (i - 1) * 4] * T[i])
                      
            else:
                # s/c propagation before the DSM
                r, v = propagate_lagrangian(
                    r_P[i], v_out, x[8 + (i - 1) * 4] * T[i] * DAY2SEC, self.common_mu)
                # Lambert arc to reach next planet in sequence during (1-nu2)*T2 (second segment)
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

        m_initial = self.rockets(x[3] / 1000., declination)
        # And we can evaluate the final mass via Tsiolkowsky
        Isp = self.Isp
        g0 = self.g0

        DVarray = self._compute_dvs(x)[0]
        DV = sum(DVarray)
        DV = DV + 165.  # losses for 3 swingbys (due to drag), have to change this to scale with the different number of flybys
        m_final = m_initial * exp(-DV / Isp / g0)
        # Numerical guard for the exponential
        if m_final == 0:
            m_final = 1e-320

        if self.constrained:
            retval = [-log(m_final), sum(T) - 3652.5]
        else:
            retval = [-log(m_final), ]
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

        # 3 - We start with the first leg
        print("First Leg: " + self._seq[0].name + " to " + self._seq[1].name)
        print("Departure: " + str(t_P[0]) +
              " (" + str(t_P[0].mjd2000) + " mjd2000) ")
        print("Duration: " + str(T[0]) + "days")
        print("VINF: " + str(x[3] / 1000) + " km/sec")

        v0 = [a + b for a, b in zip(v_P[0], [Vinfx, Vinfy, Vinfz])]
        r, v = propagate_lagrangian(
            r_P[0], v0, x[4] * T[0] * DAY2SEC, self.common_mu)

        print("DSM after " + str(x[4] * T[0]) + " days")

        # Lambert arc to reach seq[1]
        dt = (1 - x[4]) * T[0] * DAY2SEC
        l = lambert_problem_multirev(v, lambert_problem(
            r, r_P[1], dt, self.common_mu, cw=False, max_revs=self.max_revs))
        v_end_l = l.get_v2()[0]
        v_beg_l = l.get_v1()[0]

        # First DSM occuring at time nu1*T1
        DV[0] = norm([a - b for a, b in zip(v_beg_l, v)])
        print("DSM magnitude: " + str(DV[0]) + "m/s")

        # 4 - And we proceed with each successive leg
        for i in range(1, self.n_legs):
            common_mu = self.common_mu
            if self.adapt_last_leg and i == self.n_legs-1: #switching mu to be in Saturn's frame of reference
                common_mu = self._seq[i].mu_self

            print("\nleg no. " + str(i + 1) + ": " +
                  self._seq[i].name + " to " + self._seq[i + 1].name)
            print("Duration: " + str(T[i]) + "days")
            # Fly-by
            v_out = fb_prop(v_end_l, v_P[i], x[
                7 + (i - 1) * 4] * self._seq[i].radius, x[6 + (i - 1) * 4], self._seq[i].mu_self)
            print(
                "Fly-by epoch: " + str(t_P[i]) + " (" + str(t_P[i].mjd2000) + " mjd2000) ")
            print(
                "Fly-by radius: " + str(x[7 + (i - 1) * 4]) + " planetary radii")
            # s/c propagation before the DSM
            r, v = propagate_lagrangian(
                r_P[i], v_out, x[8 + (i - 1) * 4] * T[i] * DAY2SEC, self.common_mu)
            print("DSM after " + str(x[8 + (i - 1) * 4] * T[i]) + " days")
            # Lambert arc to reach Earth during (1-nu2)*T2 (second segment)
            dt = (1 - x[8 + (i - 1) * 4]) * T[i] * DAY2SEC
            l = lambert_problem_multirev(v, lambert_problem(r, r_P[i + 1], dt,
                                                            self.common_mu, cw=False, max_revs=self.max_revs))
            v_end_l = l.get_v2()[0]
            v_beg_l = l.get_v1()[0]
            # DSM occuring at time nu2*T2
            DV[i] = norm([a - b for a, b in zip(v_beg_l, v)])
            print("DSM magnitude: " + str(DV[i]) + "m/s")

        # Last Delta-v
        print("\nArrival at " + self._seq[-1].name)
        DV[-1] = norm([a - b for a, b in zip(v_end_l, v_P[-1])])
        print(
            "Arrival epoch: " + str(t_P[-1]) + " (" + str(t_P[-1].mjd2000) + " mjd2000) ")
        print("Arrival Vinf: " + str(DV[-1]) + "m/s")
        if self._orbit_insertion:
            # In this case we compute the insertion DV as a single pericenter
            # burn
            DVper = np.sqrt(DV[-1] * DV[-1] + 2 *
                            self._seq[-1].mu_self / self._rp_target)
            DVper2 = np.sqrt(2 * self._seq[-1].mu_self / self._rp_target -
                             self._seq[-1].mu_self / self._rp_target * (1. - self._e_target))
            DVinsertion = np.abs(DVper - DVper2)
            print("Insertion DV: " + str(DVinsertion) + "m/s")

        print("Total mission time: " + str(sum(T) / 365.25) + " years (" + str(sum(T)) + " days)")

        # We transform it (only the needed component) to an equatorial system rotating along x
        # (this is an approximation, assuming vernal equinox is roughly x and the ecliptic plane is roughly xy)
        earth_axis_inclination = 0.409072975
        Vinfz = - Vinfy * sin(earth_axis_inclination) + Vinfz * cos(earth_axis_inclination)
        # And we find the vinf declination (in degrees)
        sindelta = Vinfz / x[3]
        declination = asin(sindelta) / np.pi * 180.
        m_initial = self.rockets(x[3] / 1000., declination)
        # And we can evaluate the final mass via Tsiolkowsky
        Isp = self.Isp
        g0 = self.g0
        DV = sum(DV) + DVinsertion
        DV = DV + 165.  # losses for 3 swgbys + insertion
        m_final = m_initial * exp(-DV / Isp / g0)
        print("\nInitial mass:", m_initial)
        print("Final mass:", m_final)
        print("Declination:", declination)

    # Plot of the trajectory
    def plot(self, x, ax=None):
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
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from pykep.orbit_plots import plot_planet, plot_lambert, plot_kepler

        if ax is None:
            mpl.rcParams['legend.fontsize'] = 6
            fig = plt.figure()
            axis = fig.add_subplot(projection='3d')
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
                0.8, 0.6, 0.8), legend=True, units=AU, axes=axis, N=150)

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

        # First DSM occuring at time nu1*T1
        DV[0] = norm([a - b for a, b in zip(v_beg_l, v)])

        # 4 - And we proceed with each successive leg
        for i in range(1, self.n_legs):
            common_mu = self.common_mu
            if self.adapt_last_leg and i == self.n_legs - 1:  # switching mu to be in Saturn's frame of reference
                common_mu = self._seq[i].mu_self

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

            v_end_l = l.get_v2()[0]
            v_beg_l = l.get_v1()[0]
            # DSM occuring at time nu2*T2
            DV[i] = norm([a - b for a, b in zip(v_beg_l, v)])
        plt.show()
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
    planetary_sequence = [earth, venus, mars, jupiter, saturn, titan]

    # many_sequences = find_all_combinations([venus, mars, jupiter, saturn])
    # planetary_sequence = many_sequences[4]
    # planetary_sequence.insert(0, earth)
    # planetary_sequence.append(titan)

    # We solve it!!

    feasibility = False
    tries = 0
    while not feasibility:
        tries += 1
        print("Try Number {}".format(tries))
        udp = TitanChemicalUDP(sequence=planetary_sequence, adapt_last_leg=True, constrained=False)
        sol = Algorithms(problem=udp)
        #uda = sol.calculus(algo="slsqp")
        uda = sol.augmented_lagrangian(local_algo="slsqp")
        uda2 = sol.monotonic_basin_hopping(uda)
        # uda = sol.self_adaptive_differential_algorithm()
        try:
            champion = sol.archipelago(uda2, islands=8, island_population=20)
            feasibility = pg.problem(udp).feasibility_x(champion)
            if not feasibility:
                continue
            udp.pretty(champion)
            print("Feasibility = ", feasibility)

            udp.plot(champion)
            plt.show()

        except:
            continue

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
    2) Add augmented lagrangian
    
    MGA_1DSM Decision Vector:
                         0     1  2   3     4     5      6      7     8    9   ....    -1
      direct encoding: [t0] + [u, v, Vinf, eta1, T1] + [beta, rp/rV, eta2, T2] + ... 
      alpha encoding:  [t0] + [u, v, Vinf, eta1, a1] + [beta, rp/rV, eta2, a2] + ... + [T]
      eta encoding:    [t0] + [u, v, Vinf, eta1, n1] + [beta, rp/rV, eta2, n2] + ...
      
      where t0 is a mjd2000, Vinf is in km/s, T in days, beta in radians and the rest non dimensional.
      
    Implementation Steps:
    1) MGA_1DSM to Saturn only
    2) Add another variable which asks for the moons name and if you want a moon in the first place
    3) Figure out how to consider the Saturn -> Titan. Maybe do a pl2pl in pykep, but can do it using lambert or something??
    4) Append to the lagrangian and ballistic lists with the new legs
    
    """