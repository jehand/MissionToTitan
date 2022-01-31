import pykep as pk
import pygmo as pg
from chemical_propulsion import TitanChemicalUDP
from algorithms import Algorithms


def spice_kernels():
    # Downloading the spice kernel
    import os.path

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

    pk.util.load_spice_kernel("sat441.bsp")
    pk.util.load_spice_kernel("de432s.bsp")
    print("Imported SPICE kernels!")


def find_all_combinations(stuff):
    # Does not consider visiting the same planet multiple times ...
    import itertools

    combs = []
    for L in range(0, len(stuff) + 1):
        for subset in itertools.combinations(stuff, L):
            combs.append(subset)

    return combs


def run_titan_archi():
    # Redefining the planets as to change their safe radius
    earth = pk.planet.spice('EARTH BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, pk.MU_EARTH,
                            pk.EARTH_RADIUS, pk.EARTH_RADIUS * 1.05)

    venus = pk.planet.spice('VENUS BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, 100, 100, 100)
    venus.safe_radius = 1.05

    mars = pk.planet.spice('MARS BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, 100, 100, 100)
    mars.safe_radius = 1.05

    jupiter = pk.planet.spice('JUPITER BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, 100, 100, 100)
    jupiter.safe_radius = 1.7

    saturn = pk.planet.spice('SATURN BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, 100, 100, 100)
    saturn.safe_radius = 1.5

    titan = pk.planet.spice('TITAN', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, 100, 100, 100)

    # Defining the sequence and the problem
    planetary_sequence = [earth, venus, mars, jupiter, saturn, titan]
    # many_sequences = find_all_combinations([venus, mars, jupiter, saturn])
    # planetary_sequence = many_sequences[4]
    # planetary_sequence.insert(0, earth)
    # planetary_sequence.append(titan)
    udp = TitanChemicalUDP(sequence=planetary_sequence, constrained=False)

    #prob = pg.problem(udp)
    #prob.c_tol = 1e-4

    # We solve it!!
    sol = Algorithms(problem=udp)
    champion = sol.self_adaptive_differential_algorithm()
    udp.pretty(champion)
    axis = udp.plot(champion)
    axis.legend(fontsize=6)


if __name__ == "__main__":
    # Checks to make sure the spice kernels have been imported
    spice_kernels()

    # Going to create a list of functions that

    run_titan_archi()
