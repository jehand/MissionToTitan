from udps.chemical_propulsion_mk import TitanChemicalUDP
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
    for planet in planets:
        pk.orbit_plots.plot_planet(planet, t0=time, axes=ax, color="b", units=AU, legend=False)

    # Plot the position of the spacecraft
    (x,y,z), _ = eph_sc(time)
    
    x /= AU
    y /= AU
    z /= AU
    r_array.append((x, y, z))
    rs = np.array(r_array)
    ax.scatter(x, y, z, label="Spacecraft", s=10, c="pink")
    ax.scatter(0, 0, 0, label="Sun", s=10, c="yellow")
    ax.plot(rs[:,0], rs[:,1], rs[:,2], "pink")
    #ax.legend(fontsize=8)
    ax.set_zticks([])
    ax.grid(False)

    return ax

def AnimationFunction(frame, eph_sc, planets, start_time, end_time, n_frames, ax = None):
    del_time = frame * abs(end_time - start_time)/n_frames
    ax.clear()
    ax = plot_sc_and_planets(eph_sc, planets, start_time + del_time, ax)

if __name__ == "__main__":
    spice_kernels()
    venus, earth, mars, jupiter, saturn, titan = load_spice() 

    sequence = [earth, venus, venus, earth, jupiter, saturn]
    departure_range=[pk.epoch_from_string("1997-JAN-01 00:00:00.000"), pk.epoch_from_string("1997-DEC-31 00:00:00.000")]    
    
    udp = TitanChemicalMGAUDP(sequence = sequence, departure_range = departure_range)
    champion_x = [-775.6898615859894, 186.08657230116904, 400.11310660884567, 54.158500679510375, 506.5463301089962, 1299.7266803872685]
    end_time = champion_x[0] + sum(champion_x[1:])
    champion_x = times_to_eta(champion_x[0], udp.tof, champion_x[1:])
    eph = udp.get_eph_function(champion_x)
    # ax = plot_sc_and_planets(eph, sequence, pk.epoch_from_string("1999-JAN-03 00:00:00.000").mjd2000)
    # plt.show()
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

    frames = 193
    gif_name = "../cassini_validation.gif"
    
    animation = FuncAnimation(fig, 
                              AnimationFunction, 
                              frames=frames, 
                              interval=108.8, 
                              fargs=(eph, sequence, champion_x[0], end_time, frames, axis), 
                              repeat=False
                              )
    #plt.show()
    animation.save(gif_name, writer="imagemagick", fps=9.19)
    
    # video = anim_created.to_html5_video()
    # html = display.HTML(video)
    # display.display(html)
    # # good practice to close the plt object.
    # plt.close()