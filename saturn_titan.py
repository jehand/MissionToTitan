import numpy as np
import pykep as pk
from pykep.trajopt._lambert import lambert_problem_multirev
from pykep.core import epoch, DAY2SEC, MU_SUN, lambert_problem, AU, epoch
from pykep import epoch_from_string

"""
Going to define 3 functions:

1) Find Titan's current orbital plane
2) Calculate the Delta V to get into an assumed Saturn orbit
3) Calculate the lambert function from Saturn to Titan from a point in the orbit

What do I need:
1) the specific Saturn orbit we are trying for
2) Assume we do one orbit of Saturn, and then burn to Titan (so calculate how long one orbit of Saturn would be)

First attempt:
Assume we are in the same inclination plane, and calculate the lambert required
Eccentricity coming out is way too large (many there are some different units being used?)

"""

def planet_orbital_description(arrival_time, planet):
    """
    This function outputs the orbital description for a planet relative to how it was defined at a specific time.
        
    Args:
        arrival_time (``pykep.epoch``): the time at which the orbit parameters should be determined.
        planet (``pykep.planet``): the planet whose parameters would like to be measured.
    """
    
    return planet.osculating_elements(epoch(arrival_time.mjd2000, "mjd2000"))
    
def deltaV_change_inclination(incoming_inclination, target_inclination):
    # Assume the orbital change is being done when mu is the suns
    # Find the inclination angle of the spacecraft by looking at the spacecraft ephemeris data from mga_1dsm?
    # 
    
    pass

def orbit2orbit_lambert(r_starting, e_starting, target_planet, starting_time, dt, max_revs=0):
    # assume the time we arrive at the planet is that same at which we enter its orbit
    # want to propagate to the apoapsis of the orbit and then lambert transfer
    # can turn it into a pygmo problem later maybe
    # find destinations position, solve lambert problem, compute v change
    
    """
    Steps: 
    1) find the spacecrafts orbit params around Saturn
    2) propagate spacecraft to apoapsis and find the time required for that, and position + velocity at that point
    3) find Titan ephemeris at this time + dt
    4) Solve lambert problem
    5) find the change in velocity required -> ∆V
    """
    
    t_apo = 0
    
    # Finding our current orbit of saturn
    # We know our a, our e, our i, (r_p given for the orbit_insertion calculator takes the pericenter, and we also give it e)
    # Assume we are at periapsis so we know the other 2 values then
    
    
    
    r,v = 0,0
    
    dt *= DAY2SEC #converting to seconds
    r_P, v_P = target_planet.eph(epoch(starting_time.mjd2000 + dt + t_apo))
    
    l = lambert_problem_multirev(v, lambert_problem(r, r_P, dt,
                                                            target_planet.common_mu, cw=False, max_revs=max_revs))
    
    v_end_l = l.get_v2()[0]
    v_beg_l = l.get_v1()[0]
    
    
    pass
    
    
if __name__ == "__main__":
    pk.util.load_spice_kernel("sat441.bsp")
    pk.util.load_spice_kernel("de432s.bsp")
    
    # All parameters taken from: https://ssd.jpl.nasa.gov/astro_par.html
    # (and for Titan from: https://solarsystem.nasa.gov/moons/saturn-moons/titan/by-the-numbers/)
    MU_SATURN = 37940584.841800 * 1e9
    MU_TITAN = 8980.14303 * 1e9

    # All parameters taken from: https://solarsystem.nasa.gov/resources/686/solar-system-sizes/
    # (and for Titan from: https://solarsystem.nasa.gov/moons/saturn-moons/titan/by-the-numbers/)
    R_SATURN = 58232 * 1e3
    R_TITAN = 2574.7 * 1e3

    # Spice has arguments: target, observer, ref_frame, abberations, mu_central_body, mu_self, radius, safe_radius
    earth = pk.planet.spice('EARTH BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, pk.MU_EARTH,
                            pk.EARTH_RADIUS, pk.EARTH_RADIUS * 1.05)

    saturn = pk.planet.spice('SATURN BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_SATURN,
                             R_SATURN, R_SATURN)
    saturn.safe_radius = 1.5

    #Defining Titan relative to Saturn instead
    titan = pk.planet.spice('TITAN', 'SATURN BARYCENTER', 'ECLIPJ2000', 'NONE', MU_SATURN, MU_TITAN,
                            R_TITAN, R_TITAN)
    
    time = epoch_from_string("2021-DEC-28 11:58:50.816")
    print(planet_orbital_description(time,titan))