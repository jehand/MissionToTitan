import numpy as np
import pykep as pk
import pygmo as pg
from pykep.core import epoch, DAY2YEAR
from  chemical_propulsion2 import TitanChemicalUDP
from algorithms import Algorithms
from planetary_system import PlanetToSatellite
import matplotlib.pyplot as plt
import matplotlib
import os.path

matplotlib.use('Qt5Agg')
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class ScrollableWindow(QtWidgets.QMainWindow):
    def __init__(self, fig):
        self.qapp = QtWidgets.QApplication([])

        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(0,0,0,0)
        self.widget.layout().setSpacing(0)

        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.scroll = QtWidgets.QScrollArea(self.widget)
        self.scroll.setWidget(self.canvas)

        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.scroll)

        self.showMaximized()
        exit(self.qapp.exec_()) 

def spice_kernels():
    # Downloading the spice kernel
    if not os.path.exists("sat441.bsp") or not os.path.exists("de432s.bsp"):
        import requests

        url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat441.bsp"
        r = requests.get(url, allow_redirects=True)
        open('sat441.bsp', 'wb').write(r.content)

        print("Downloaded sat441.bsp!")

        url2 = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de432s.bsp"
        r2 = requests.get(url2, allow_redirects=True)
        open('de432s.bsp', 'wb').write(r2.content)

        print("Downloaded de432s.bsp!")

    else:
        print("File is already downloaded!")

    pk.util.load_spice_kernel("de432s.bsp")
    pk.util.load_spice_kernel("sat441.bsp")
    print("Imported SPICE kernels!")

def load_spice():
    # All parameters taken from: https://ssd.jpl.nasa.gov/astro_par.html
    # (and for Titan from: https://solarsystem.nasa.gov/moons/saturn-moons/titan/by-the-numbers/)
    MU_VENUS = 324858.592000 * 1e9
    MU_MARS = 42828.375816 * 1e9
    MU_JUPITER = 126712764.100000 * 1e9
    MU_SATURN = 37940584.841800 * 1e9
    MU_TITAN = 8980.14303 * 1e9

    # All parameters taken from: https://solarsystem.nasa.gov/resources/686/solar-system-sizes/
    # (and for Titan from: https://solarsystem.nasa.gov/moons/saturn-moons/titan/by-the-numbers/)
    R_VENUS = 6052 * 1e3
    R_MARS = 3390 * 1e3
    R_JUPITER = 69911 * 1e3
    R_SATURN = 58232 * 1e3
    R_TITAN = 2574.7 * 1e3

    # Spice has arguments: target, observer, ref_frame, abberations, mu_central_body, mu_self, radius, safe_radius
    earth = pk.planet.spice('EARTH BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, pk.MU_EARTH,
                            pk.EARTH_RADIUS, pk.EARTH_RADIUS * 1.1)
    earth.name = "EARTH"

    venus = pk.planet.spice('VENUS BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_VENUS,
                            R_VENUS, R_VENUS*1.1)
    venus.name = "VENUS"

    mars = pk.planet.spice('MARS BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_MARS,
                           R_MARS, R_MARS*1.1)
    mars.name = "MARS"

    jupiter = pk.planet.spice('JUPITER BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_JUPITER,
                              R_JUPITER, R_JUPITER*1.1)
    jupiter.name = "JUPITER"
   

    saturn = pk.planet.spice('SATURN BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_SATURN,
                             R_SATURN, R_SATURN*1.1)
    saturn.name = "SATURN"
    
    #Defining Titan relative to Saturn instead
    titan = pk.planet.spice('TITAN', 'SATURN BARYCENTER', 'ECLIPJ2000', 'NONE', MU_SATURN, MU_TITAN,
                            R_TITAN, R_TITAN)
    titan.name = "TITAN"
    
    return venus, earth, mars, jupiter, saturn, titan

class TrajectorySolver():
    def __init__(self, interplanetary_udp=None, planetary_udp=None):
        """
        This class is used to solve for a single trajectory or to run a DoE on the sequences. 
        
        Args:
            interplanetary_udp (``pygmo.problem``)   : the pygmo problem object for the interplanetary phase
            planetary_udp (``pygmo.problem``)        : the pygmo problem object for the planetary phase
        """        
        # Saving the user defined problems
        self.interplanetary_problem = interplanetary_udp
        self.planetary_problem = planetary_udp

    def interplanetary_trajectory(self, sequence, departure_range):
        """
        This class is used to solve for a single trajectory or to run a DoE on the sequences. 
        
        Args:
            sequence (``array(pykep.planet)``)       : the sequence of planets to visit in the interplanetary phase
            target_satellite (``pykep.planet``)      : the final satellite (moon or asteroid) to visit after the last planet in sequence 
                                                    (this planets ephemeris data should be given relative to the last planet in 
                                                    the sequence)
            departure_range (``array(pykep.epoch)``) : the range of dates for which to begin the interplanetary stage
            target_orbit (``array(double)``)         : the final orbit desired at the satellite given as [r_target, e_target] where 
                                                    r_target is in m.
        """
        # Define the interplanetary problem
        interplanetary_udp = self.interplanetary_problem(sequence) ##### Need to add departure date etc.
        self.interplanetary_udp = interplanetary_udp
        self.departure_range = departure_range
        self.sequence = sequence
          
        # We solve it!!
        uda = pg.nlopt('bobyqa') 
        uda.ftol_rel = 1e-12
        uda = pg.algorithm(uda)
        uda.set_verbosity(100)
        archi = pg.archipelago(algo=uda, prob=interplanetary_udp, n=8, pop_size=20)

        archi.evolve(20)
        archi.wait()
        sols = archi.get_champions_f()
        sols2 = [item[0] for item in sols]
        idx = sols2.index(min(sols2))
        DV = sols2[idx]
        champion_interplanetary = archi.get_champions_x()[idx]
        
        return champion_interplanetary, DV
    
    def compute_final_vinf(self, final_planet, champion_interplanetary):
        """
        Compute the arrival v_inf at the final planet in the sequence.

        Args:
            final_planet (``pykep.planet``)             : the final planet in the interplanetary sequence
            champion_interplanetary (``array(double)``) : the decision chromosome for the interplanetary phase.
        """
        
        # Take the decision chromosome from the interplanetary phase as inputs for the Saturnian system
        DV, lamberts, T, ballistic_legs, ballistic_ep = self.interplanetary_udp._compute_dvs(champion_interplanetary)
        
        # Time reaching final interplanetary planet as a pykep epoch
        t = epoch(champion_interplanetary[0] + sum(T))
        
        # Velocity at final interplanetary planet
        v_sc_solarsystem = lamberts[-1].get_v2()[0] # sc velocity relative to solar system
        _, v_P_solarsystem = final_planet.eph(t)
        v_sc = [a-b for a,b in zip(v_sc_solarsystem, v_P_solarsystem)] # sc velocity relative to last planet
        
        return v_sc, t        
    
    def planetary_trajectory(self, starting_planet, target_satellite, starting_time, target_orbit, v_sc=None):
        """
        Calculates the planetary trajectory.

        Args:
            starting_planet (``pykep.planet``)  : the starting planet.
            target_satellite (``pykep.planet``) : the target destination.
            starting_time (``pykep.epoch``)     : the starting time at which the spacecraft is at periapsis around the starting planet.
            target_orbit (``array(double)``)    : the target orbit desired at the satellite as [periapsis radius, eccentricity]
            v_sc (``array(double)``)            : defines the initial velocity of the spacecraft before the starting orbit; defaults to None to 
                                                  say there's no incoming velocity.
        """
        # Solve the destination planet system to reach the satellite
        if v_sc is None:
            planetary_udp = self.planetary_problem(starting_time, target_orbit[0], target_orbit[1], starting_planet, target_satellite, 
                                                tof=[10,100], r_start_max=15, initial_insertion=False, v_inf=v_sc, max_revs=5)
        else:
            planetary_udp = self.planetary_problem(starting_time, target_orbit[0], target_orbit[1], starting_planet, target_satellite, 
                                                tof=[10,100], r_start_max=15, initial_insertion=True, v_inf=v_sc, max_revs=5)
        
        self.planetary_udp = planetary_udp
        # We solve it!!
        uda = pg.nlopt('bobyqa') 
        uda.ftol_rel = 1e-12
        uda = pg.algorithm(uda)
        uda.set_verbosity(100)
        archi = pg.archipelago(algo=uda, prob=planetary_udp, n=8, pop_size=20)

        archi.evolve(20)
        archi.wait()
        sols = archi.get_champions_f()
        sols2 = [item[0] for item in sols]
        idx = sols2.index(min(sols2))
        DV = sols2[idx]
        champion_planetary = archi.get_champions_x()[idx]
        
        return champion_planetary, DV
        
    def pretty(self, champion_interplanetary=None, champion_planetary=None):
        """
        Write results in human readable format for the planetary and interplanetary part

        Args:
            champion_interplanetary (``array``) : the solution decision chromosome for the interplanetary sequence
            champion_planetary (``array``)      : the solution decision chromosome for the planetary sequence
        """
        
        # Write out pretty results
        if list(champion_interplanetary):
            print("==================================")
            print("Interplanetary Stage:")
            print("==================================")
            self.interplanetary_udp.pretty(champion_interplanetary)
            print("")
        
        if list(champion_planetary):
            print("==================================")
            print("Planetary Stage:")
            print("==================================")
            self.planetary_udp.pretty(champion_planetary)
            print("")
            
        if list(champion_interplanetary) and list(champion_interplanetary): # if we are doing the entire trajectory
            print("==================================")
            print("Total mission:")
            print("==================================")
            DV, t_departure, t_arrival, _, T = self.get_results(champion_interplanetary, champion_planetary)
            
            print("Total DV = {0:.4g}".format(DV/1000), "km/s")
            print("Departure Date = {}".format(t_departure))
            print("Total Flight Time = {0:.4g}".format(T*DAY2YEAR), "years ({0:.7g}".format(T), "days)")
            print("Arrival Date = {}".format(t_arrival))
            print("\n")
    
    def plot(self, champ_interplanetary=None, champ_planetary=None):
        """
        Plot results for the planetary and interplanetary part

        Args:
            champ_interplanetary (``array``) : the solution decision chromosome for the interplanetary sequence
            champ_planetary (``array``)      : the solution decision chromosome for the planetary sequence
        """
        fig = plt.figure(figsize=(10,10))
        plt.rc('legend',fontsize=6) # using a size in points

        if list(champ_interplanetary) and list(champ_planetary) is None: # plot just interplanetary
            ax = fig.add_subplot(projection='3d')
            self.interplanetary_udp.plot(champ_interplanetary)

        elif list(champ_planetary) and list(champ_interplanetary) is None: # plot just planetary
            ax = fig.add_subplot(projection='3d')
            self.planetary_udp.plot(champ_planetary)
            
        else: # plot both together
            ax2 = fig.add_subplot(3, 2, 2, projection='3d')
            ax2.set_title("Planetary Sequence", fontsize=18)
            self.planetary_udp.plot(champ_planetary, ax=ax2)
            ax2.legend(loc='center left', bbox_to_anchor=(1.07, 0.5))
            
            ax4 = fig.add_subplot(3, 2, 4, projection="3d")
            ax4.view_init(elev=90, azim=0)
            self.planetary_udp.plot(champ_planetary, ax=ax4)
            ax4.get_legend().remove()
            
            ax6 = fig.add_subplot(3, 2, 6, projection="3d")
            ax6.view_init(elev=0, azim=0)
            self.planetary_udp.plot(champ_planetary, ax=ax6)
            ax6.get_legend().remove()

            ax1 = fig.add_subplot(3, 2, 1, projection='3d')
            ax1.set_title("Interplanetary Sequence", fontsize=18)
            self.interplanetary_udp.plot(champ_interplanetary, ax=ax1)
            ax1.legend(loc='center left', bbox_to_anchor=(1.07, 0.5))

            ax3 = fig.add_subplot(3, 2, 3, projection='3d')
            ax3.view_init(elev=90, azim=0)
            self.interplanetary_udp.plot(champ_interplanetary, ax=ax3)
            ax3.get_legend().remove()

            ax5 = fig.add_subplot(3, 2, 5, projection='3d')
            ax5.view_init(elev=0, azim=0)
            self.interplanetary_udp.plot(champ_interplanetary, ax=ax5)
            ax5.get_legend().remove()
        
        #fig.subplots_adjust(wspace=0, hspace=0)
        ScrollableWindow(fig)
        #plt.show()
        
    def get_results(self, champ_interplanetary, champ_planetary):
        """
        Get the results for the entire journey.

        Args:
            champ_interplanetary (``array``) : the solution decision chromosome for the interplanetary sequence
            champ_planetary (``array``)      : the solution decision chromosome for the planetary sequence
        """
        # Give overall results for both legs
        DV = self.interplanetary_udp.fitness(champ_interplanetary)[0]
        DV += self.planetary_udp.fitness(champ_planetary)[0]
        
        t_departure = champ_interplanetary[0]
                
        _, _, T, _, _ = self.interplanetary_udp._compute_dvs(champ_interplanetary)
        tof = sum(T) + champ_planetary[-1]
        t = tof + t_departure
        
        return DV, epoch(t_departure), epoch(t), [sum(T), champ_planetary[-1]], tof

    def entire_trajectory(self, sequence, departure_dates, target_satellite, target_orbit):
        champ_inter, _ = self.interplanetary_trajectory(sequence=sequence, departure_range=departure_dates)
        v_sc, start_time = self.compute_final_vinf(sequence[-1], champ_inter)
        champ_plan, _ = self.planetary_trajectory(sequence[-1], target_satellite, start_time, target_orbit, v_sc)
        
        return champ_inter, champ_plan

    # def extract_seq_from_csv(self, filepath, planets, starting_planet, ending_planet):
    #     """
    #     Converts a .csv file with sequences given as a pattern (underneath the column name Pattern) into 
    #     an array of pykep.planet objects.
        
    #     E.g. 
    #     planets = {1 : earth, 2 : venus, 3 : saturn}
    #     starting_planet = mercury
    #     ending_planet = uranus
    #     Given a pattern 213 from the .csv, the outcome is a mercury --> venus --> earth --> saturn --> uranus

    #     Args:
    #         filepath (``string``)               : the filepath of the .csv with pattern information
    #         planets (``dic(int:pykep.planet)``) : a dictionary mapping the numbers in the pattern to planets.
    #         starting_planet (``pykep.planet``)  : the initial planet in the sequence (always appended to the front of pattern)
    #         ending_planet (``pykep.planet``)    : the final planet in the sequence (always appended to the end of pattern)
    #     """
        
    #     patterns = np.genfromtxt(filepath, delimiter=",", names=True)["Pattern"]
        
        
    # def run_doe(self, DoE_filepath, output_filepath, interplanetary_planets, 
    #             departure_dates, target_satellite, target_orbit):
        
    #     # Each leg:
    #     champ_inter, inter_DV = self.interplanetary_trajectory(sequence=interplanetary_sequence, departure_range=departure_dates)
    #     v_sc, start_time = self.compute_final_vinf(interplanetary_sequence[-1], champ_inter)
    #     champ_plan, plan_DV = self.planetary_trajectory(interplanetary_sequence[-1], target_satellite, start_time, target_orbit, v_sc)
        
    #     # Output the decision chromosomes, the DV for each part, the total DV, the time for each part, the total time.
        
        
if __name__ == "__main__":
    # Checks to make sure the spice kernels have been imported
    spice_kernels()
    venus, earth, mars, jupiter, saturn, titan = load_spice()
    
    sequence = [earth, saturn]
    target_satellite = titan
    departure_dates = []
    target_orbit = [titan.radius * 2, 0.1]
    
    trajectory = TrajectorySolver(TitanChemicalUDP, PlanetToSatellite)
    
    champ_inter, champ_plan = trajectory.entire_trajectory(sequence=sequence, departure_dates=departure_dates,
                                                           target_satellite=titan,
                                                           target_orbit=target_orbit)

    champ_inter = [10790.386060785193, 0.06204046642376808, 0.561252533215833, 10000.0, 0.01, 1500.0]
    champ_plan = [15.0, 0.17972278837262365, 0.09095129656224789, 21.57452055244928]
    
    trajectory.pretty(champ_inter, champ_plan)
    trajectory.plot(champ_inter, champ_plan)
