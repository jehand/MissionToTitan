import pykep as pk
import numpy as np
from trajectory_solver import spice_kernels, load_spice
from pykep import epoch_from_string
from pykep.core import epoch, AU
from scipy import interpolate
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm

"""
Read in our spice data and create the planet objects. Then extract the ephemeris in some time grid, then interpolate between those grids
"""

def norm(a1, a2):
    # Calculate the norm of the difference between 2 arrays a1 and a2
    return np.sqrt(np.sum([(a-b)**2 for a,b in zip(a1,a2)]))

def sample_el(planet, time_range, t_res):
    """
    Given a pykep.planet object and a time range, determine the ephemeris data at based on t_res.

    Args:
        planet (``pykep.planet``): the planet object for which to sample its ephemeris data
        time_range (``array(pykep.epoch)``): the time range for which to do the sampling
        t_res (``int``): number of days between each point
    """
    times = np.arange(time_range[0].mjd2000, time_range[1].mjd2000, t_res)
    if time_range[1].mjd2000 not in times:
        times = np.append(times, time_range[1].mjd2000)

    elements = np.empty(shape=(len(times), 6))
    for i in range(len(times)):
        r,v = planet.eph(epoch(times[i], "mjd2000"))
        elements[i] = pk.ic2par(r,v,planet.mu_central_body)

    return elements, times

def el_interp(elements, times, type="linear"):
    """
    Find the ephemeris functions at specific times by interpolation.

    Args:
        elements (``array(double)``): array of 6 osculating elements at different points in time
        times (``array(double)``): array of the times at which each vector was determined in mjd
        type (``str``) : the type of interpolation to do using scipy
    """    
    funcs = []
    for i in range(3):
         funcs.append(interpolate.interp1d(times, elements[:,i], kind=type))

    # For W and w, the angles go from [0,2π]. So we need to account for the sudden jump in the interpolation
    # Assume that there are no jumps greater than π suddenly, so we are not assuming monotonically increasing
    for i in range(3,5):
        funcs.append(interpolate.interp1d(times, np.unwrap(elements[:,i]), kind=type))

    # For the mean true anomaly, because it cycles between [-π,π], we need to properly account for that in how we do the interpolation
    # We make the assumption that it is monotonically increasing so we can account for >π jumps.
    M = np.array(elements[:,5]) + np.pi
    jumps = np.diff(M)<0 # Assuming that orbits are all going in the same direction
    M_nojumps = np.hstack((M[0], M[1:] + np.cumsum(jumps) * 2 * np.pi))
    funcs.append(interpolate.interp1d(times, M_nojumps, kind=type))
    
    return funcs

def create_planet(planet, time_window, time_res):
    """
    This function creates a new instance of the class new_planet

    Args:
        planet (``pykep.planet``): the planet object to interpolate
        time_window (``array(pykep.epochh)``): the window within which to interpolate
        time_res (``float``): the time step between each measurement
    """
    elements_array, t = sample_el(planet, time_window, time_res)
    interps = el_interp(elements_array, t, kind="linear")
    planet_new = new_planet(planet, interps)
    
    return planet_new

def save(obj, out_filename):
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)

    with open(out_filename, "wb") as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
    print("Saved planet: {}!".format(obj.name))
    
class new_planet(pk.planet._base):
    def __init__(self, planet, interps):
        """
        This class creates a new planet object that does not rely on SPICE using interpolation

        Args:
            planet (``pykep.planet``): the planet object to interpolate
            interps (``array(scipy.interp1d)``): the array of interpolations for the osculating elements
        """
        super().__init__(
            planet.mu_central_body,
            planet.mu_self,
            planet.radius,
            planet.safe_radius,
            planet.name  
        )
        self.planet = planet
        self.interpolations = interps
    
    def __getinitargs__(self):
        # So that we can save with pickle later
        return self.planet, self.interpolations
    
    def eph(self, when):
        time = when.mjd2000
        elements = np.empty(6)
        for i in range(6):
            elements[i] = self.interpolations[i](time)
        
        for i in range(3,6):
            elements[i] %= 2*np.pi
        elements[5] -= np.pi     
          
        return pk.par2ic(elements, self.mu_central_body)
    
    def osculating_elements(self, when):
        r,v = self.eph(when)
        results = pk.ic2par(r,v,self.mu_central_body)
            
        return results
    
    
if __name__ == "__main__":
    spice_kernels()
    venus, earth, mars, jupiter, saturn, titan = load_spice()
    
    window = ["1990-JAN-01 00:00:00.000", "2050-JAN-02 00:00:00.000"]
    time_window = [epoch_from_string(window[0]), epoch_from_string(window[1])]
    planet = saturn
    step = 1
    out_filename = "planets/"+planet.name + window[0][:4] + "-" + window[1][:4] + "_" + str(step) + ".pkl"   

    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    first_time = time_window[0].mjd2000
    last_time = time_window[1].mjd2000
    N = 200000
    t = [np.random.random()*(last_time-first_time) + first_time for _ in range(N)]
    
    if (True):
        with open(out_filename, "rb") as inp:
            planet_new = pickle.load(inp)
    else:
        elements_array, t_sample = sample_el(planet, time_window, step)
        interps = el_interp(elements_array, t_sample, type="cubic")
        planet_new = new_planet(planet, interps)
        save(planet_new, out_filename)
    
    if (False):
        # Plotting an orbit comparison
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        pk.orbit_plots.plot_planet(planet_new, t0=time_window[0], tf=time_window[1], N=2000, axes = ax, color='b', units=AU, legend=True)
        pk.orbit_plots.plot_planet(planet, t0=time_window[0], tf=time_window[1], N=2000, axes = ax, color='k', units=AU, legend=True)
        plt.draw()
        
    if (False):
        print("\n Ephemeris Errors...")
        # Plotting the elements over time
        print("Finding new orbital elements...")
        els = np.array([planet_new.osculating_elements(epoch(time, "mjd2000")) for time in tqdm(t, miniters=0)])
        print("Finding SPICE orbital elements")
        elements = np.array([planet.osculating_elements(epoch(time, "mjd2000")) for time in tqdm(t, miniters=0)])
        
        t, els, elements = zip(*sorted(zip(t,els,elements)))
        elements = np.array(elements)
        els = np.array(els)
        a_new = elements[:,0]
        a = els[:,0]
        a_errors = a_new - a
        print("a errors = ", np.std(a_errors, ddof=1))
        
        e_new = elements[:,1]
        e = els[:,1]
        e_errors = e_new - e
        print("e errors = ", np.std(e_errors, ddof=1))

        i_new = elements[:,2]
        i = els[:,2]
        i_errors = i_new - i
        print("i errors = ", np.std(i_errors, ddof=1))

        W_new = elements[:,3]
        W = els[:,3]
        W_errors = W_new - W
        print("W errors = ", np.std(W_errors, ddof=1))

        w_new = elements[:,4]
        w = els[:,4]
        w_errors = w_new - w
        print("w errors = ", np.std(w_errors, ddof=1))

        M_new = elements[:,5]
        M = els[:,5]
        M_errors = M_new - M
        print("M errors = ", np.std(M_errors, ddof=1))
        
        fig, ax = plt.subplots(3, 2, figsize=(12,8))
        ax[0,0].plot(t, a_new, "k-", label="SPICE")
        ax[0,0].plot(t, a, "r--", label="Interpolation")
        ax[0,0].set_ylabel("a")
        ax[0,0].set_xlabel("Time (mjd2000)")
        ax[0,0].legend()
     
        ax[0,1].plot(t, e_new, "k-", label="SPICE")
        ax[0,1].plot(t, e, "r--", label="Interpolation")
        ax[0,1].set_ylabel("e")
        ax[0,1].set_xlabel("Time (mjd2000)")
        ax[0,1].legend()
           
        ax[1,0].plot(t, i_new, "k-", label="SPICE")
        ax[1,0].plot(t, i, "r--", label="Interpolation")
        ax[1,0].set_ylabel("i")
        ax[1,0].set_xlabel("Time (mjd2000)")
        ax[1,0].legend()
        
        ax[1,1].plot(t, W_new, "k-", label="SPICE")
        ax[1,1].plot(t, W, "r--", label="Interpolation")
        ax[1,1].set_ylabel(r"$\Omega$")
        ax[1,1].set_xlabel("Time (mjd2000)")
        ax[1,1].legend()

        ax[2,0].plot(t, w_new, "k-", label="SPICE")
        ax[2,0].plot(t, w, "r--", label="Interpolation")
        ax[2,0].set_ylabel(r"$\omega$")
        ax[2,0].set_xlabel("Time (mjd2000)")
        ax[2,0].legend()
        
        ax[2,1].plot(t, M_new, "k-", label="SPICE")
        ax[2,1].plot(t, M, "r--", label="Interpolation")
        ax[2,1].set_ylabel("M")
        ax[2,1].set_xlabel("Time (mjd2000)")
        ax[2,1].legend()
            
        plt.tight_layout()
        plt.savefig("planets/"+planet_new.name + window[0][:4] + "-" + window[1][:4] + "_" + str(step) + "el.png", bbox_inches="tight", dpi=400)
        plt.draw()

    if (False):
        # Plotting errors between our guess and the actual results
        print("\n Ephemeris Errors...")
        first_time = time_window[0].mjd2000
        last_time = time_window[1].mjd2000
        N = 10000
        t = [np.random.random()*(last_time-first_time) + first_time for _ in range(N)]
            
        r_errors = np.empty(len(t))
        v_errors = np.empty(len(t))
        for i in tqdm(range(len(r_errors)), miniters=0):
            r_errors[i] = norm(planet_new.eph(epoch(t[i],"mjd2000"))[0], planet.eph(epoch(t[i],"mjd2000"))[0])#/norm(planet.eph(epoch(t[i],"mjd2000"))[0],[0,0,0])
            v_errors[i] = norm(planet_new.eph(epoch(t[i],"mjd2000"))[1], planet.eph(epoch(t[i],"mjd2000"))[1])#/norm(planet.eph(epoch(t[i],"mjd2000"))[1],[0,0,0])

        print("r VAR =", np.std(r_errors, ddof=1))
        print("v VAR =", np.std(v_errors, ddof=1))

        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12,8))
        ts, rs, vs = zip(*sorted(zip(t,r_errors,v_errors)))
        ax1.plot(ts, rs)
        ax1.set_ylabel("r errors")
        ax1.set_xlabel("Time (mjd2000)")

        ax2.plot(ts, vs)
        ax2.set_ylabel("v errors")
        ax2.set_xlabel("Time (mjd2000)")
        
        plt.tight_layout()
        plt.savefig("planets/"+planet_new.name + window[0][:4] + "-" + window[1][:4] + "_" + str(step) + "eph.png", bbox_inches="tight", dpi=400)
        plt.show()