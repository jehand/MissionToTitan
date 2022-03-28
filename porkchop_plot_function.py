"""
Porkchop Plotter
AEON Grand Challenge
Spring 2022
Sarah Hopkins
"""

# General Imports
import pykep as pk
import pygmo as pg
from datetime import datetime as dt
import numpy as np

# Plotting imports
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

'''
#######################################################################################################################
Porkchop Plotter Function
#######################################################################################################################
'''
def porkchop_plotter(T0, Vinf_dep, e_target, tof_bounds, n, t_step):
    # T0:               launch date
    # planet_sequence:  sequence of planets for fly-bys
    # Vinf_dep:         dv leaving initial planet (km/s)
    # e_target:         insertion orbit eccentricity
    # tof_bounds:       window for entire trajectory - days
    # n:                number of iterations
    # t_step:           time step (days)

    pk.util.load_spice_kernel('DE423.bsp')
    earth = pk.planet.spice(
        'EARTH',  # 'target_id', body of interest
        'SUN',  # 'observer_id', the center of the reference frame
        'ECLIPJ2000',  # reference frame, point of origin for calculations
        'NONE',  # abberations; corrections accounting for finite light speed for observations; unnecessary
        pk.MU_SUN,  # Mu of the central body for reference frame (Mu = "standard gravitational parameter" = G*M
        pk.MU_EARTH,  # Mu of target body
        6378100.,  # radius of target body (meters)
        6378100. * 1.1  # safe radius for target body (meters)
    )
    earth.name = 'EARTH'

    venus = pk.planet.spice(
        'VENUS',
        'SUN',
        'ECLIPJ2000',
        'NONE',
        pk.MU_SUN,
        3.24859e14,  # Mu of target body, Venus
        6051800.,  # Radius of target body (meters), Venus
        6051800. * 1.1  # Safe radius for target body (meters), Venus
    )
    venus.name = 'VENUS'

    mars = pk.planet.spice(
        'MARS',
        'SUN',
        'ECLIPJ2000',
        'NONE',
        pk.MU_SUN,
        4.282837e13,  # Mu of target body, Mars
        3389500.,  # Radius of target body (meters), Mars
        3389500 * 1.1  # Safe radius for target body (meters), Mars
    )
    mars.name = 'MARS'

    jupiter = pk.planet.spice(
        'JUPITER BARYCENTER',
        'SUN',
        'ECLIPJ2000',
        'NONE',
        pk.MU_SUN,
        1.26686534e+17,  # Mu of target body, Jupiter
        69911000.,  # Radius of target body (meters), Jupiter
        69911000. * 1.02  # Safe radius for target body (meters), Jupiter
    )
    jupiter.name = 'JUPITER'

    saturn = pk.planet.spice(
        'Saturn BARYCENTER',
        'SUN',
        'ECLIPJ2000',
        'NONE',
        pk.MU_SUN,
        3.7931187e+16,  # Mu of the target body, Saturn
        58232000.,  # Radius of target body (meters), Saturn
        58232000. * 1.02  # Safe radius for target body (meters), Saturn
    )
    saturn.name = 'SATURN'

    # Common parameters
    multi_objective = False     # single objective for min dV
    orbit_insertion = True      # are you inserting at the end? Yes (True)
    rp_target = jupiter.safe_radius  # orbit insertion radius of periapsis, m (defined in dictionary created above)

    # Alpha Transcription MGA, no variation in departure time, single objective
    encoding = 'alpha'  # change 'alpha' to 'direct' if you want to use the direct method

    alpha_mga = pk.trajopt.mga(
        seq=planet_sequence,
        t0=[T0, pk.epoch(T0.mjd2000 + T0_u)],
        tof=tof_bounds,
        vinf=Vinf_dep,
        multi_objective=multi_objective,
        tof_encoding=encoding,
        orbit_insertion=orbit_insertion,
        e_target=e_target,
        rp_target=rp_target
    )

    def mga_dV(t0: type(pk.epoch(0)), tof: float):
        mga_udp = pk.trajopt.mga(
            seq=planet_sequence,
            t0=[t0, t0],
            tof=[tof, tof],
            vinf=Vinf_dep,
            multi_objective=False,
            tof_encoding="alpha",
            orbit_insertion=orbit_insertion,
            e_target=e_target,
            rp_target=rp_target,
        )

        # declare problem (and tolerances)
        prob = pg.problem(mga_udp)  # mga = multi-gravity assist; udp = user datagram protocol
        # algorithm setup; each library of solvers uses different setups. This one demonstrates NLopt setup.
        # Nlopt = nonlinear optimization
        nl_setup = pg.nlopt("bobyqa")  # bobyqa = finds the max of a function
        nl_setup.xtol_rel = 1e-6  # xtol_rel = a fractional tolerance on the parameters x
        # nl_setup.maxeval = 1500
        algo = pg.algorithm(nl_setup)
        # set up individuals to evolve
        pop = pg.population(prob, 200)

        # perform optimization until either stopping criteria is first reached
        pop = algo.evolve(pop)

        # retrieve data
        alpha_delta_v = pop.get_f()[pop.best_idx()]
        __, *best_tofs = alpha_mga.alpha2direct(pop.get_x()[pop.best_idx()])

        return alpha_delta_v, best_tofs

    # Create sweeps and storage
    # n = 6                         # number of iterations
    dt_departure = np.empty([n, ])  # creates an empty matrix based on n rows
    # t_step = 30.0                   # time step - days
    dt_arrival = np.linspace(100, 3000, n, True)  # 100 = 100 days post launch; 3000 = 3000 days post launch;n = # steps
    dVs = np.empty([n, n])
    dT1 = np.empty([n, n])
    dT2 = np.empty([n, n])

    for i in range(0, n):
        t0 = pk.epoch(T0.mjd2000 + i * t_step)  # "MJD2000" = modified Julian date
        dt_departure[i] = t0.mjd2000
        print("Iteration {} of {}".format(i + 1, n))
        for j, tf in enumerate(dt_arrival):
            dv, ti = mga_dV(t0, tf)  # could sub this out for function "entire trajectory" in chemical propulsion file
            dVs[i][j] = dv
            dT1[i][j] = ti[0]
            dT2[i][j] = ti[1]
            dVs[i][j] = dv
            dT1[i][j] = ti[0]
            dT2[i][j] = ti[1]

    # Plot
    fig1 = plt.figure(figsize=(30, 30))
    plt.contour(dVs, 5)
    plt.xlabel("Earth Launch Date")
    plt.ylabel("Titan Arrival Date")
    plt.title("Earth to Titan Porkchop Plot")
    clb = plt.colorbar()
    clb.set_label('Total DeltaV (km/s)')

    plt.show()


    return()

'''
#######################################################################################################################
Calling and Using the Porkchop Plotter Function
#######################################################################################################################
'''

T0 = pk.epoch_from_string("2027-January-01 12:00:00")           # Launch date
planet_sequence = [earth, mars, venus, mars, jupiter, saturn]   # start at Earth, end at Jupiter
Vinf_dep = 2.                 # km/s, dv leaving initial planet
e_target = 0.75               # orbit insertion eccentricity, ND
T0_u = 0                      # upper bound on departure time
tof_bounds = [50, 6000]       # window for entire trajectory - days
n = 6                         # number of iterations
t_step = 30.0                 # time step - days


# put the call to the function in a loop so we create multiple porkchop plots at once

porkchop_plotter()


'''
#######################################################################################################################
'''



