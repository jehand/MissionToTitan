import pykep as pk
from pykep.trajopt import lt_margo, mr_lt_nep, launchers
from pykep.planet import jpl_lp
from pykep import epoch_from_string
from algorithms import Algorithms
import matplotlib.pyplot as plt

import numpy as np
from math import log, acos, cos, sin, asin, exp

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

class MGAElectricalPropulsion(mr_lt_nep):
    """
    This class uses a multiple rendezvous (MR) approach to reach Titan modelled with electrical propulsion (meant to be solar). 
    For solar, the power is calculated to decay with distance from the sun (not including impacting objects that might get in the way). 
    The trajectory is modelled using the Sims-Flanagan model.

    .. note::

       The idea is to use a MR approach and set the time at each planet to be minimal so that .
    """
    def __init__(self):
        pass

if __name__ == "__main__":
    pk.util.load_spice_kernel("sat427.bsp")
    pk.util.load_spice_kernel("de432s.bsp")
    mars = pk.planet.spice('MARS BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, 100, 100, 100)
    mars.safe_radius = 1.05
    
    udp = P2PElectricalPropulsion(mars)
    sol = Algorithms(problem=udp)
    champion = sol.self_adaptive_differential_algorithm()
    udp.pretty(champion)
    axis = udp.plot_traj(champion)
    axis.legend(fontsize=6)
    plt.show()
    
