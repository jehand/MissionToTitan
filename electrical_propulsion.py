import pykep as pk
from pykep.trajopt import lt_margo, launchers
from pykep.planet import jpl_lp
from pykep import epoch_from_string
from algorithms import Algorithms
import matplotlib.pyplot as plt

import numpy as np
from math import log, acos, cos, sin, asin, exp

# Going to try a lt_margo implementation (idea being that the low thrust model would be used to reach Mars, and then from there 
# chemical would be used, therefore we do not need to model the initial trajectory as a MGA?)

class ElectricalPropulsion(lt_margo):
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

        Args: #Needs editing
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


    # def fitness(self, x):
    #     T, Vinfx, Vinfy, Vinfz = self._decode_times_and_vinf(x)
    #     # We transform it (only the needed component) to an equatorial system rotating along x
    #     # (this is an approximation, assuming vernal equinox is roughly x and the ecliptic plane is roughly xy)
    #     earth_axis_inclination = 0.409072975
    #     # This is different from the GTOP tanmEM problem, I think it was bugged there as the rotation was in the wrong direction.
    #     Vinfz = - Vinfy * sin(earth_axis_inclination) + Vinfz * cos(earth_axis_inclination)
    #     # And we find the vinf declination (in degrees)
    #     sindelta = Vinfz / x[3]
    #     declination = asin(sindelta) / np.pi * 180.
    #     # We now have the initial mass of the spacecraft
    #     m_initial = launchers.atlas501(x[3] / 1000., declination) # Need to change the launcher model being used parametrically
    #     # And we can evaluate the final mass via Tsiolkowsky
    #     Isp = 312.
    #     g0 = 9.80665
    #     DV = super().fitness(x)[0]
    #     DV = DV + 165.  # losses for 3 swgbys + insertion
    #     m_final = m_initial * exp(-DV / Isp / g0)
    #     # Numerical guard for the exponential
    #     if m_final == 0:
    #         m_final = 1e-320
    #     if self.constrained:
    #         retval = [-log(m_final), sum(T) - 3652.5]
    #     else:
    #         retval = [-log(m_final), ]
    #     return retval

    # def pretty(self, x):
    #     """
    #     prob.plot(x)

    #     - x: encoded trajectory

    #     Prints human readable information on the trajectory represented by the decision vector x

    #     Example::

    #       print(prob.pretty(x))
    #     """
    #     super().pretty(x)
    #     T, Vinfx, Vinfy, Vinfz = self._decode_times_and_vinf(x)
    #     # We transform it (only the needed component) to an equatorial system rotating along x
    #     # (this is an approximation, assuming vernal equinox is roughly x and the ecliptic plane is roughly xy)
    #     earth_axis_inclination = 0.409072975
    #     Vinfz = - Vinfy * sin(earth_axis_inclination) + Vinfz * cos(earth_axis_inclination)
    #     # And we find the vinf declination (in degrees)
    #     sindelta = Vinfz / x[3]
    #     declination = asin(sindelta) / np.pi * 180.
    #     m_initial = launchers.soyuzf(x[3] / 1000., declination)
    #     # And we can evaluate the final mass via Tsiolkowsky
    #     Isp = 312.
    #     g0 = 9.80665
    #     DV = super().fitness(x)[0]
    #     DV = DV + 165.  # losses for 3 swgbys + insertion
    #     m_final = m_initial * exp(-DV / Isp / g0)
    #     print("\nInitial mass:", m_initial)
    #     print("Final mass:", m_final)
    #     print("Declination:", declination)

    def __repr__(self):
        return "AEON low thrust trajectory optimization from Earth to another planet"


if __name__ == "__main__":
    pk.util.load_spice_kernel("sat427.bsp")
    pk.util.load_spice_kernel("de432s.bsp")
    mars = pk.planet.spice('MARS BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, 100, 100, 100)
    mars.safe_radius = 1.05
    
    udp = ElectricalPropulsion(mars)
    sol = Algorithms(problem=udp)
    champion = sol.self_adaptive_differential_algorithm()
    udp.pretty(champion)
    axis = udp.plot_traj(champion)
    axis.legend(fontsize=6)
    plt.show()
    