import pykep as pk
from pykep.trajopt import lt_margo, mga_lt_nep, mr_lt_nep, launchers
from pykep.planet import jpl_lp
from pykep.core import epoch, DAY2SEC, MU_SUN, lambert_problem, propagate_lagrangian, fb_prop, AU, EARTH_VELOCITY
from pykep import epoch_from_string
from pykep.trajopt._lambert import lambert_problem_multirev
from algorithms import Algorithms
import matplotlib.pyplot as plt

import numpy as np
from math import log, acos, cos, sin, asin, exp, pi, sqrt

# Avoiding scipy dependency
def norm(x):
    return sqrt(sum([it * it for it in x]))

# Going to try a lt_margo implementation (idea being that the low thrust model would be used to reach Mars, and then from there 
# chemical would be used, therefore we do not need to model the initial trajectory as a MGA?)

# Could add a launcher model to not need an initial mass, just have the model calculate the initial mass

class P2PElectricalPropulsion(lt_margo):
    """
    This class represents a rendezvous mission to a nearby planet modelled with electrical propulsion (can be 
    solar or nuclear as determined by the user). For solar, the power is calculated to decay with distance from 
    the sun (not including impacting objects that might get in the way). The trajectory is modelled using the 
    Sims-Flanagan model, extended to include the Earth's gravity (assumed constant on each segment).

    .. note::

       The idea is to have electrical go to Mars, and then use chemical after. Hence, this is not being modelled
       as a MGA, but just as a direct trajectory to Mars (or really any planet you input).
    """

    def __init__(self, destination, departure_time=["2021-DEC-28 11:58:50.816", "2027-DEC-28 11:58:50.816"], initial_mass=20.0, 
                 thrust=0.0017, I_sp=3000.0, solar=True):
        """
        Defining the problem and allowing the user to input most of the parameters

        Args:
            - destination (``pykep.planet``): The target planet for the trajectory.
            - departure_time (``array(str, str)``): The window within which the launch may happen.
            - initial_mass (``float``): The initial mass of the spacecraft.
            - thrust (``float``): The thrust of the propulsion system at 1AU.
            - Isp (``float``): The specific impulse of the propulsion system at 1AU. 
            - solar (``bool``): Activates a solar thrust model for the thrust - distance dependency.
        """

        super().__init__(
            target=destination,
            n_seg=30,
            grid_type="uniform",
            t0=[pk.epoch_from_string(departure_time[0]), pk.epoch_from_string(departure_time[1])],
            tof=[200, 365.25 * 4],
            m0=initial_mass,
            Tmax=thrust,
            Isp=I_sp,
            earth_gravity=False,
            sep=solar,
            start="earth"
        )

    def __repr__(self):
        return "AEON low thrust trajectory optimization from Earth to another planet"


class MGAElectricalPropulsion(mga_lt_nep):
    """
    This class uses a multiple gravity assist (MGA) approach to reach Titan modelled with electrical propulsion (meant to be solar). 
    For solar, the power is calculated to decay with distance from the sun (not including impacting objects that might get in the way). 
    The trajectory is modelled using the Sims-Flanagan model.

    .. note::

       The idea is to use the regular MGA_LT class in pykep, but add way to do solar power.
    """
    def __init__(self, sequence, departure_time=["2021-DEC-28 11:58:50.816", "2027-DEC-28 11:58:50.816"], initial_mass=[1200., 2000.0], 
                 thrust=0.5, I_sp=3500.0, solar=True, high_fidelity_analysis=False):
        """
        Defining the problem and allowing the user to input most of the parameters

        Args:
            - sequence (``array`` of ``pykep.planet``): An ordered list of the flybys for the calculated trajectory.
            - departure_time (``2D-array`` of ``str``): The window within which the launch may happen.
            - initial_mass (``2D-array`` of ``float``): The range of initial masses for the spacecraft.
            - thrust (``float``): The thrust of the propulsion system at 1AU.
            - Isp (``float``): The specific impulse of the propulsion system at 1AU. 
            - solar (``bool``): Activates a solar thrust model for the thrust - distance dependency.
            - high_fidelity_analysis (``bool``): Makes the trajectory computations slower, but actually dynamically feasible.
        """

        super().__init__(
            seq = sequence,
            n_seg = [100 for _ in range(len(sequence)-1)],
            t0 = [pk.epoch_from_string(departure_time[0]).mjd2000, pk.epoch_from_string(departure_time[1]).mjd2000],
            tof = [[100, 2000] for _ in range(len(sequence)-1)],
            vinf_dep = 3, #Need to change
            vinf_arr = 2, #Need to change
            mass = initial_mass,
            Tmax = thrust,
            Isp = I_sp,
            fb_rel_vel = 6,
            multi_objective = False,
            high_fidelity = high_fidelity_analysis
        )

    def pretty(self, x):
        """
        prob.pretty(x)
        - x: encoded trajectory
        Prints human readable information on the trajectory represented by the decision vector x
        Example::
          print(prob.pretty(x))
        """
        from pykep.sims_flanagan import sc_state, leg
        from pykep.core import EARTH_VELOCITY

        # We compute the epochs and ephemerides of the planetary encounters
        t_P = list([None] * (self._n_legs + 1))
        r_P = list([None] * (self._n_legs + 1))
        v_P = list([None] * (self._n_legs + 1))
        for i in range(len(self._seq)):
            t_P[i] = epoch(x[0] + sum(x[1:i * 8:8]))
            r_P[i], v_P[i] = self._seq[i].eph(t_P[i])

        T = []
        for i in range(self._n_legs):
            T.append(t_P[i+1].mjd2000 - t_P[i].mjd2000)

        print("First Leg: " + self._seq[0].name + " to " + self._seq[1].name)
        print("Departure: " + str(t_P[0]) +
              " (" + str(t_P[0].mjd2000) + " mjd2000) ")
        print("Duration: " + str(T[0]) + "days")
        print("VINF: " + str(x[3] * x[3] + x[4] * x[4] + x[5] * x[5] / (EARTH_VELOCITY * EARTH_VELOCITY)) + " km/sec") # Check this

        # We assemble the constraints.
        # 1 - Mismatch Constraints
        for i in range(1, self._n_legs):
            # Departure velocity of the spacecraft in the heliocentric frame
            print("\nleg no. " + str(i + 1) + ": " +
                  self._seq[i].name + " to " + self._seq[i + 1].name)
            print("Duration: " + str(T[i]) + "days")
            print(
                "Fly-by epoch: " + str(t_P[i+1]) + " (" + str(t_P[i+1].mjd2000) + " mjd2000) ")
            print(
                "Fly-by radius: " + str(x[2 + (i) * 8]) + " planetary radii") # Check this

            v0 = [a + b for a, b in zip(v_P[i], x[3 + 8 * i:6 + 8 * i])]
            if i==0:
                m0 = self._mass[1]
            else:
                m0 = x[2 + 8 * (i-1)]
            x0 = sc_state(r_P[i], v0, m0)
            vf = [a + b for a, b in zip(v_P[i+1], x[6 + 8 * i:9 + 8 * i])]
            xf = sc_state(r_P[i+1], vf, x[2 + 8 * i])
            idx_start = 1 + 8 * self._n_legs + sum(self._n_seg[:i]) * 3
            idx_end   = 1 + 8 * self._n_legs + sum(self._n_seg[:i+1]) * 3
            self._leg.set(t_P[i], x0, x[idx_start:idx_end], t_P[i+1], xf)
            #print(self._leg)
            times, r, v, m = self._leg.get_states()
            #print("Radius", r[-1])

        print("\nArrival at " + self._seq[-1].name)
        print(
            "Arrival epoch: " + str(t_P[-1]) + " (" + str(t_P[-1].mjd2000) + " mjd2000) ")

        n_fb = self._n_legs - 1
        v_arr = (x[6 + n_fb * 8] * x[6 + n_fb * 8] + x[7 + n_fb * 8] * x[7 + n_fb * 8] + x[8 + n_fb * 8] * x[
            8 + n_fb * 8])
        print("Arrival Vinf: " + str(v_arr) + "m/s")

        print("Total mission time: " + str(sum(T) / 365.25) + " years (" + str(sum(T)) + " days)")


    def __repr__(self):
        return "AEON low thrust MGA trajectory optimization"


class MRElectricalPropulsion(mr_lt_nep):
    """
    This class uses a multiple rendevous (MR) approach to reach Titan modelled with electrical propulsion (meant to be solar). 
    For solar, the power is calculated to decay with distance from the sun (not including impacting objects that might get in the way). 
    The trajectory is modelled using the Sims-Flanagan model.

    .. note::

       The idea is to use the regular MR_LT class in pykep, but add way to do solar power.
    """
    def __init__(self, sequence, departure_time=["2021-DEC-28 11:58:50.816", "2027-DEC-28 11:58:50.816"], initial_mass=[1200., 2000.0], 
                 thrust=0.5, I_sp=3500.0, solar=True, high_fidelity_analysis=False):
        """
        Defining the problem and allowing the user to input most of the parameters

        Args:
            - sequence (``array`` of ``pykep.planet``): An ordered list of the flybys for the calculated trajectory.
            - departure_time (``2D-array`` of ``str``): The window within which the launch may happen.
            - initial_mass (``2D-array`` of ``float``): The range of initial masses for the spacecraft.
            - thrust (``float``): The thrust of the propulsion system at 1AU.
            - Isp (``float``): The specific impulse of the propulsion system at 1AU. 
            - solar (``bool``): Activates a solar thrust model for the thrust - distance dependency.
            - high_fidelity_analysis (``bool``): Makes the trajectory computations slower, but actually dynamically feasible.
        """

        super().__init__(
            seq = sequence,
            n_seg = [20 for _ in range(len(sequence)-1)],
            t0 = [pk.epoch_from_string(departure_time[0]).mjd2000, pk.epoch_from_string(departure_time[1]).mjd2000],
            tof = [[100, 2000] for _ in range(len(sequence)-1)],
            vinf_dep = 3, #Need to change
            vinf_arr = 2, #Need to change
            mass = initial_mass,
            Tmax = thrust,
            Isp = I_sp,
            fb_rel_vel = 6,
            multi_objective = False,
            high_fidelity = high_fidelity_analysis
        )

if __name__ == "__main__":
    pk.util.load_spice_kernel("sat441.bsp")
    pk.util.load_spice_kernel("de432s.bsp")

    # Running the P2P electrical example
    # mars = pk.planet.spice('MARS BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, 100, 100, 100)
    # mars.safe_radius = 1.05
    #
    # udp = P2PElectricalPropulsion(mars)
    # sol = Algorithms(problem=udp)
    # champion = sol.self_adaptive_differential_algorithm()
    # udp.pretty(champion)
    # axis = udp.plot_traj(champion)
    # axis.legend(fontsize=6)
    # plt.show()
    # Running the MGA electrical example
    earth = pk.planet.spice('EARTH BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, pk.MU_EARTH,
                            pk.EARTH_RADIUS, pk.EARTH_RADIUS * 1.05)

    venus = pk.planet.spice('VENUS BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, 100, 100, 100)
    venus.safe_radius = 1.05

    mars = pk.planet.spice('MARS BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, 100, 100, 100)
    mars.safe_radius = 1.05

    jupiter = pk.planet.spice('JUPITER BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, 100, 100, 100)
    jupiter.safe_radius = 1.7

    saturn = pk.planet.spice('SATURN BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, 100, 100, 100)
    saturn.safe_radius = 1.5

    titan = pk.planet.spice('TITAN', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, 100, 100, 100)

    # Defining the sequence and the problem
    planetary_sequence = [earth, mars, jupiter]
    udp = MGAElectricalPropulsion(planetary_sequence, high_fidelity_analysis=True)
    sol = Algorithms(problem=udp)
    champion = sol.self_adaptive_differential_algorithm()
    udp.pretty(champion)
    axis = udp.plot(champion)
    #axis.legend(fontsize=6)
    plt.show()