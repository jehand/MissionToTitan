from udps.chemical_mga import TitanChemicalMGAUDP
from udps.planetary_system import PlanetToSatellite
from trajectory_solver import TrajectorySolver, load_spice, spice_kernels
from ast import literal_eval
import pygmo as pg
from pykep.planet import jpl_lp
import matplotlib.pyplot as plt
import numpy as np
import pykep as pk
from pykep.trajopt import mga
from pykep.core import AU
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from IPython import display

def times_to_eta(t0, t0fmax, leg_times):
    # Decision chromosome: [t0, n1, n2, n3, ...]
    x = [t0]
    T = []

    for leg in leg_times:
        eta = (leg)/(t0fmax - np.sum(T))
        x.append(eta)
        T.append(leg)

    return x

r_array = []
def plot_sc_and_planets(eph_sc, planets, time, ax = None):
    """
    Args:
        eph_sc (``pykep ephemeris function``): the ephemeris function of the spacecraft
        planets (``array (pykep.planets)``): the planets to plot at the same time as plotting the spacecraft
        time (``float``): the mjd2000 time at which to get the ephemeris data
    """

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    
    # Plot the planets
    venus, earth, mars, jupiter, saturn = planets
    # for planet in planets:
    #     pk.orbit_plots.plot_planet(planet, t0=time, axes=ax, color="green", s=10, units=AU, legend=False)

    pk.orbit_plots.plot_planet(venus, t0=time, axes=ax, color="aqua", s=15, units=AU, legend=False)
    pk.orbit_plots.plot_planet(earth, t0=time, axes=ax, color="dodgerblue", s=15, units=AU, legend=False)
    pk.orbit_plots.plot_planet(mars, t0=time, axes=ax, color="red", s=15, units=AU, legend=False)
    pk.orbit_plots.plot_planet(jupiter, t0=time, axes=ax, color="orange", s=15, units=AU, legend=False)
    pk.orbit_plots.plot_planet(saturn, t0=time, axes=ax, color="limegreen", s=15, units=AU, legend=False)

    # Plot the position of the spacecraft
    (x,y,z), _ = eph_sc(time)
    
    x /= AU
    y /= AU
    z /= AU
    r_array.append((x, y, z))
    rs = np.array(r_array)
    ax.scatter(x, y, z, label="Spacecraft", s=10, c="fuchsia")
    ax.scatter(0, 0, 0, label="Sun", s=10, c="yellow")
    ax.plot(rs[:,0], rs[:,1], rs[:,2], "fuchsia")
    #ax.legend(fontsize=8)
    ax.set_zticks([])
    ax.grid(False)

    return ax

def f_log_decor(orig_fitness_function):
    def new_fitness_function(self, dv):
        if hasattr(self, "dv_log"):
            self.dv_log.append(dv)
        else:
            self.dv_log = [dv]
        return orig_fitness_function(self, dv)
    return new_fitness_function

def AnimationFunction(frame, eph_sc, planets, start_time, end_time, n_frames, ax = None):
    print("Frame {}".format(frame + 1))
    del_time = frame * abs(end_time - start_time)/n_frames
    ax.clear()
    ax = plot_sc_and_planets(eph_sc, planets, start_time + del_time, ax)
    #ax.text2D(0.33, 0.95, str(pk.epoch(start_time + del_time))[:-16], transform=ax.transAxes, fontsize=14)
    ax.set_xlim(-2,10)
    ax.set_ylim(-2,10)        
    #ax.set_xlim(-2.2,2.2)
    #ax.set_ylim(-2.2,2.2)        
    
def AnimationFunctionAlgorithmEvolution(frame, udp, champion_history, ax = None):
    print("Frame {}".format(frame + 1))
    champion = champion_history[0][frame]
    dv = champion_history[1][frame]
    ax.clear()
    ax = udp.plot(champion, ax=ax)
    ax.get_legend().remove()
    ax.set_zticks([])
    ax.grid(False)
    ax.text2D(0.15, 0.95, "DV = {0:.4g} km/s".format(dv/1000) + ", Gen = {}".format(frame+1), transform=ax.transAxes, fontsize=14)

if __name__ == "__main__":
    spice_kernels()
    venus, earth, mars, jupiter, saturn, titan = load_spice() 
    sequence = [earth, venus, venus, earth, jupiter, saturn]
    departure_range=[pk.epoch_from_string("1997-JAN-01 00:00:00.000"), pk.epoch_from_string("1997-DEC-31 00:00:00.000")]    
    udp = TitanChemicalMGAUDP(sequence = sequence, departure_range = departure_range, tof=3000, tof_encoding="eta")
    
    plt.style.use('dark_background')
    fig = plt.figure()
    axis = fig.add_subplot(projection='3d')
    fig.set_facecolor('black')
    axis.set_facecolor('black')
    axis.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    axis.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    axis.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    
    axis.view_init(elev=90, azim=0)
    axis.set_zticks([])
    axis.grid(False)
    
    frames = 750
    gif_name = "../cassini.gif" #"../EVEES_inner.gif"
    
    if (True):   # [11376.982292324155, 0.12544638656734067, 0.13719950326921726, 0.4336895276521287, 0.9989999999612518]
        champion_x =  [-775.6898615859894, 186.08657230116904, 400.11310660884567, 54.158500679510375, 506.5463301089962, 1299.7266803872685] # Cassini
        #udp.pretty(champion_x)
        #direct = udp.eta2direct(champion_x)
        end_time = champion_x[0] + sum(champion_x[1:])#sum(direct)
        champion_x = times_to_eta(champion_x[0], udp.tof, champion_x[1:])
        eph = udp.get_eph_function(champion_x)
        #udp = pg.problem(pg.decorator_problem(udp, fitness_decorator=f_log_decor))
        animation = FuncAnimation(fig,
                            AnimationFunction, 
                            frames=frames, 
                            interval=16.67, 
                            fargs=(eph, [venus, earth, mars, jupiter, saturn], champion_x[0], end_time, frames, axis), 
                            repeat=False
                            )
    
    if (False):
        glob = pg.algorithm(pg.gaco(gen=1))
        pop_num = 100
        pop = pg.population(prob=udp,size=pop_num)
        gens = frames
        dvariables = []
        fitness = []
        for i in range(gens):
            pop = glob.evolve(pop)
            dvariables.append(pop.champion_x)
            fitness.append(pop.champion_f[0])

        dvariables = np.array(dvariables)
        fitness = np.array(fitness)
        animation = FuncAnimation(fig, 
                                AnimationFunctionAlgorithmEvolution, 
                                frames=frames, 
                                interval=10, 
                                fargs=(udp, (dvariables, fitness), axis), 
                                repeat=False
                                )
    
    #plt.show()
    r_array = []
    animation.save(gif_name, writer="imagemagick", fps=60, dpi=300)