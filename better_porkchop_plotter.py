# Porkchop Plotter with Stephanie's Help
# AEON Grand Challenge
# Spring 2022

import pykep as pk
import pygmo as pg
from datetime import datetime as dt
import numpy as np
from random import random as rn

# Plotting imports
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Use datetime to get today's date as starting point
T0 = pk.epoch_from_string(dt.today().isoformat().replace('T',' '))

pk.util.load_spice_kernel('DE423.bsp')
earth = pk.planet.spice(
    'EARTH',
    'SUN',
    'ECLIPJ2000',
    'NONE',
    pk.MU_SUN,
    pk.MU_EARTH,
    6378000.,
    6378000. * 1.1
)
earth.name = 'EARTH'

venus = pk.planet.spice(
    'VENUS',
    'SUN',
    'ECLIPJ2000',
    'NONE',
    pk.MU_SUN,
    3.24859e14,
    6657200.,
    6657200. * 1.1
)
venus.name = 'VENUS'

mars = pk.planet.spice(
    'MARS',
    'SUN',
    'ECLIPJ2000',
    'NONE',
    pk.MU_SUN,
    4.2828e13,
    3397000.,
    3397000 * 1.1
)
mars.name = 'MARS'

jupiter = pk.planet.spice(
    'JUPITER BARYCENTER',
    'SUN',
    'ECLIPJ2000',
    'NONE',
    pk.MU_SUN,
    1.26686534e+17,
    71492000.,
    643428000. * 1.02
)
jupiter.name = 'JUPITER'

# Common parameters
planet_sequence = [earth, mars, venus, mars, jupiter]  # start at Earth, end at Mars
Vinf_dep = 2.  # km/s
multi_objective = False  # single objective for min dV
orbit_insertion = True  # insert at the end?
e_target = 0.75  # orbit insertion eccentricity, ND
rp_target = jupiter.safe_radius # orbit insertion radius of periapsis, m

# Alpha Transcription MGA, no variation in departure time, single objective
T0_u = 0  # upper bound on departure time
tof_bounds = [50, 6000]  # window for entire trajectory
encoding = 'alpha'

alpha_mga = pk.trajopt.mga(
    seq=planet_sequence,
    t0=[T0,pk.epoch(T0.mjd2000+T0_u)],
    tof=tof_bounds,
    vinf=Vinf_dep,
    multi_objective=multi_objective,
    tof_encoding=encoding,
    orbit_insertion=orbit_insertion,
    e_target=e_target,
    rp_target=rp_target
)

# create function for t0, tof in; dv, tof_i out.
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
    prob = pg.problem(mga_udp)
    # algorithm setup; each library of solvers uses different setups. This one will demonstrate NLopt setup.
    nl_setup = pg.nlopt("bobyqa")
    nl_setup.xtol_rel = 1e-6
    # nl_setup.maxeval = 1500
    algo = pg.algorithm(nl_setup)
    # set up individuals to evolve
    pop = pg.population(prob, 200)

    # perform the optimization until either stopping criteria is first reached
    pop = algo.evolve(pop)

    # retrieve data
    alpha_delta_v = pop.get_f()[pop.best_idx()]
    __, *best_tofs = alpha_mga.alpha2direct(pop.get_x()[pop.best_idx()])
    return alpha_delta_v, best_tofs

# create sweeps and storage
n = 5
dt_departure = np.empty([n,])
t_step = 30.0
dt_arrival = np.linspace(100, 3000, n, True)
dVs = np.empty([n, n])
dT1 = np.empty([n, n])
dT2 = np.empty([n, n])

for i in range(0, n):
    t0 = pk.epoch(T0.mjd2000 + i * t_step)
    dt_departure[i] = t0.mjd2000
    print("Iteration {} of {}".format(i+1, n))
    for j, tf in enumerate(dt_arrival):
        dv, ti = mga_dV(t0, tf)
        dVs[i][j] = dv
        dT1[i][j] = ti[0]
        dT2[i][j] = ti[1]
        dVs[i][j] = dv
        dT1[i][j] = ti[0]
        dT2[i][j] = ti[1]

# Plot
fig1 = plt.figure(figsize=(30, 30))
plt.contourf(dVs, 500)
plt.xlabel("")
plt.ylabel("", fontsize=14)
plt.title("")
plt.colorbar()


plt.show()
