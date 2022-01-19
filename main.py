import pykep as pk
import pygmo as pg
from chemical_propulsion import _tandem_udp


def spice_kernels():
    # Downloading the spice kernel
    import os.path

    if not os.path.exists("sat427.bsp") or not os.path.exists("de432s.bsp"):
        import requests

        url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat427.bsp"
        r = requests.get(url, allow_redirects=True)
        open('sat427.bsp', 'wb').write(r.content)

        print("Downloaded sat427.bsp!")

        url2 = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de432s.bsp"
        r2 = requests.get(url2, allow_redirects=True)
        open('de432s.bsp', 'wb').write(r2.content)

        print("Downloaded de432s.bsp!")

    else:
        print("File is already downloaded!")

    pk.util.load_spice_kernel("sat427.bsp")
    pk.util.load_spice_kernel("de432s.bsp")
    print("Imported SPICE kernels!")


def run_tandem():
    # seq = [jpl_lp('earth'), jpl_lp('venus'), jpl_lp('earth')]
    udp = _tandem_udp(prob_id=12)

    prob = pg.problem(udp)
    # We solve it!!
    uda = pg.sade(gen=100)
    islands = 8
    archi = pg.archipelago(algo=uda, prob=udp, n=islands, pop_size=20)
    print(
        "Running a Self-Adaptive Differential Evolution Algorithm .... on {} parallel islands".format(islands))
    archi.evolve(10)
    archi.wait()
    sols = archi.get_champions_f()
    sols2 = [item[0] for item in sols]
    idx = sols2.index(min(sols2))
    print("Done!! Solutions found are: ", archi.get_champions_f())
    udp.pretty(archi.get_champions_x()[idx])
    axis = udp.plot(archi.get_champions_x()[idx])
    axis.legend(fontsize=6)


if __name__ == "__main__":
    spice_kernels()
    run_tandem()
