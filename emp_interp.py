import pykep as pk
from pykep.core import DAY2YEAR
import csv
from datetime import datetime as dt
import numpy as np
from trajectory_solver import load_spice, spice_kernels
import os

def main(body, t_step, t_final):
    T0 = pk.epoch_from_string(dt.today().isoformat().replace("T", " "))
    total = int(t_final/t_step)
    keys = ["mjd2000", "a", "e", "i", "W", "w", "M"]
    ephm = []
    ME = []
    for j in range(total):
        t = pk.epoch(T0.mjd2000 + j * t_step)
        temp = (t.mjd2000, ) + body.osculating_elements(t)
        ME.append(temp[-1])
        ephm.append({key:temp[i] for i, key in enumerate(keys)})

    ME_edit = np.unwrap(ME)
    for i, j in enumerate(ME_edit):
        ephm[i][keys[-1]] = j

    out_filename = "interpolated_data/"+body.name+".csv"
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    with open(out_filename, "w+", newline="") as csv_f:
        writer = csv.DictWriter(
            csv_f, keys)
        writer.writeheader()

        for entry in ephm:
            writer.writerow(entry)

    print("Saved file for {}!".format(body.name))

if __name__ == "__main__":
    spice_kernels()
    planets = load_spice()
    years = 25
    days = years*1/DAY2YEAR

    for planet in planets:
        main(planet, 0.1, days)