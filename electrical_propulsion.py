import pykep as pk
from pykep.trajopt import lt_margo, mga_lt_nep, mr_lt_nep, launchers
from pykep.planet import jpl_lp
from pykep.core import epoch, AU
from pykep import epoch_from_string
from pykep.sims_flanagan import sc_state

from algorithms import Algorithms
import matplotlib.pyplot as plt
import pygmo as pg

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
    This class uses a multiple gravity assist (MGA) approach to reach Titan modelled with electrical propulsion (where
    Solar is modelled just as the lowest thrust option). The trajectory is modelled using the Sims-Flanagan model.

    The decision vector (chromosome) is::
      [t0] + 
      [T1, mf1, Vxi1, Vyi1, Vzi1, Vxf1, Vyf1, Vzf1] + 
      [T2, mf2, Vxi2, Vyi2, Vzi2, Vxf2, Vyf2, Vzf2] + 
      ... + 
      [throttles1] + 
      [throttles2] + 
      ...

    .. note::

       The idea is to use the regular MGA_LT class in pykep, but add way to do solar power.
    """
    def __init__(self, sequence, departure_time=["2021-DEC-28 11:58:50.816", "2027-DEC-28 11:58:50.816"], initial_mass=[1200., 2000.0], 
                 thrust=0.5, I_sp=3500.0, high_fidelity_analysis=False):
        """
        Defining the problem and allowing the user to input most of the parameters

        Args:
            - sequence (``array`` of ``pykep.planet``): An ordered list of the flybys for the calculated trajectory.
            - departure_time (``2D-array`` of ``str``): The window within which the launch may happen.
            - initial_mass (``2D-array`` of ``float``): The range of initial masses for the spacecraft.
            - thrust (``float``): The thrust of the propulsion system at 1AU.
            - Isp (``float``): The specific impulse of the propulsion system at 1AU. 
            - high_fidelity_analysis (``bool``): Makes the trajectory computations slower, but actually dynamically feasible.
        """

        super().__init__(
            seq = sequence,
            n_seg = [5 for _ in range(len(sequence)-1)],
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

    # And this helps visualizing the trajectory
    def plot(self, x, axes=None):
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
        from pykep.orbit_plots import plot_sf_leg, plot_planet

        # Creating the axis if necessary
        if axes is None:
            mpl.rcParams['legend.fontsize'] = 10
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        else:
            ax = axes

        # Plotting the Sun ........
        ax.scatter([0], [0], [0], color=['y'])

        # We compute the epochs and ephemerides of the planetary encounters
        t_P = list([None] * (self._n_legs + 1))
        r_P = list([None] * (self._n_legs + 1))
        v_P = list([None] * (self._n_legs + 1))
        for i in range(len(self._seq)):
            t_P[i] = epoch(x[0] + sum(x[1:i*8:8]))
            r_P[i], v_P[i] = self._seq[i].eph(t_P[i])
            plot_planet(self._seq[i], t0 = t_P[i],
                        units=AU, legend=True, color=(0.7, 0.7, 0.7), s=30, axes=ax)

        # We assemble the constraints.
        # 1 - Mismatch Constraints
        for i in range(self._n_legs):
            # Departure velocity of the spacecraft in the heliocentric frame
            v0 = [a + b for a, b in zip(v_P[i], x[3 + 8 * i:6 + 8 * i])]
            if i == 0:
                m0 = self._mass[1]
            else:
                m0 = x[2 + 8 * (i-1)]
            x0 = sc_state(r_P[i], v0, m0)
            vf = [a + b for a, b in zip(v_P[i+1], x[6 + 8 * i:9 + 8 * i])]
            xf = sc_state(r_P[i+1], vf, x[2 + 8 * i])
            idx_start = 1 + 8 * self._n_legs + sum(self._n_seg[:i]) * 3
            idx_end   = 1 + 8 * self._n_legs + sum(self._n_seg[:i+1]) * 3
            self._leg.set(t_P[i], x0, x[idx_start:idx_end], t_P[i+1], xf)
            plot_sf_leg(self._leg, units=AU, N=10, axes=ax, legend=False)

        return ax

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
        print("Duration: " + str(T[0]) + " days")

        v_inf = [x[3], x[4], x[5]]
        v_first_f = [x[6], x[7], x[8]]
        print("VINF: " + str(norm(v_inf)/1000) + " km/sec")

        DV = [norm([a - b for a, b in zip(v_first_f, v_inf)])]
        print("Delta V: " + str(DV[0]) + " m/s")
        # Text for the different legs
        for i in range(1, self._n_legs):
            # Departure velocity of the spacecraft in the heliocentric frame
            print("\nleg no. " + str(i + 1) + ": " +
                  self._seq[i].name + " to " + self._seq[i + 1].name)
            print("Duration: " + str(T[i]) + "days")
            print(
                "Fly-by epoch: " + str(t_P[i]) + " (" + str(t_P[i].mjd2000) + " mjd2000) ")

            print("Mass change: " + str(x[2 + 8*(i-1)] - x[2 + 8*i]) + " kg")

            v0 = [a + b for a, b in zip(v_P[i], x[3 + 8 * i:6 + 8 * i])]
            vf = [a + b for a, b in zip(v_P[i+1], x[6 + 8 * i:9 + 8 * i])]

            delta_v = norm([a-b for a,b in zip(vf,v0)])
            DV.append(delta_v)
            print("Delta V: " + str(delta_v) + " m/s")

        print("\nArrival at " + self._seq[-1].name)
        print(
            "Arrival epoch: " + str(t_P[-1]) + " (" + str(t_P[-1].mjd2000) + " mjd2000) ")

        n_fb = self._n_legs - 1
        v_arr = sqrt(x[6 + n_fb * 8] * x[6 + n_fb * 8] + x[7 + n_fb * 8] * x[7 + n_fb * 8] + x[8 + n_fb * 8] * x[
            8 + n_fb * 8])
        print("Arrival Vinf: " + str(v_arr/1000) + " km/s")

        m_0 = x[2]
        m_f = x[2 + n_fb * 8]

        print("\n" + "Initial Mass: " + str(m_0) + " kg")
        print("Final Mass: " + str(m_f) + " kg")
        print("Total Delta V: " + str(sum(DV)) + " m/s")
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

    # Defining the sequence and solving the optimization problem
    planetary_sequence = [earth, mars, jupiter]
    udp = MGAElectricalPropulsion(planetary_sequence, high_fidelity_analysis=True)
    sol = Algorithms(problem=udp)
    champion = sol.self_adaptive_differential_algorithm()

    print("Feasible: ", pg.problem(udp).feasibility_x(champion))
    udp.pretty(champion)
    axis = udp.plot(champion)
    axis.legend(fontsize=6)
    plt.show()
