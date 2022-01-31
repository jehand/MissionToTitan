import pykep as pk
from pykep.trajopt import mga_1dsm, launchers
from pykep.planet import jpl_lp
from pykep import epoch_from_string

import numpy as np
from math import log, acos, cos, sin, asin, exp


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

    def __init__(self, sequence=1, constrained=True):
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
            t0=[pk.epoch_from_string("2021-DEC-28 11:58:50.816"), pk.epoch_from_string("2027-DEC-28 11:58:50.816")],
            tof=[[20, 1000] for _ in range(len(sequence)-1)],
            vinf=[2.5, 4.9],
            add_vinf_dep=False,
            add_vinf_arr=True,
            tof_encoding='direct',
            multi_objective=False,
            orbit_insertion=True,
            e_target=0.98531407996358,
            rp_target=80330000,
            eta_lb=0.01,
            eta_ub=0.99,
            rp_ub=10
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
        m_initial = launchers.atlas501(x[3] / 1000., declination) # Need to change the launcher model being used parametrically
        # And we can evaluate the final mass via Tsiolkowsky
        Isp = 312.
        g0 = 9.80665
        DV = super().fitness(x)[0]
        DV = DV + 165.  # losses for 3 swgbys + insertion
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
        return "TandEM sequence: " + str(self.sequence)

    def get_extra_info(self):
        retval = "\t Sequence: " + \
                 [pl.name for pl in self._seq].__repr__() + "\n\t Constrained: " + \
                 str(self.constrained)
        return retval

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
        m_initial = launchers.soyuzf(x[3] / 1000., declination)
        # And we can evaluate the final mass via Tsiolkowsky
        Isp = 312.
        g0 = 9.80665
        DV = super().fitness(x)[0]
        DV = DV + 165.  # losses for 3 swgbys + insertion
        m_final = m_initial * exp(-DV / Isp / g0)
        print("\nInitial mass:", m_initial)
        print("Final mass:", m_final)
        print("Declination:", declination)

    def __repr__(self):
        return "AEON (Trajectory Optimisation for a Rendezvous with Titan)"

