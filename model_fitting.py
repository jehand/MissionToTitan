import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# Import the csv data
planet_name = "EARTH"
data = np.genfromtxt("interpolated_data/{}.csv".format(planet_name), names=True, delimiter=",")
t = data["mjd2000"]
a = data["a"]

def model(time, a1, c1, d1, a2, c2, d2, a3, c3, d3, i):
    return (a1*np.sin(c1*time + d1)) + (a2*np.sin(c2*time + d2)) + (a2*np.sin(c2*time + d2)) + i - np.mean(a)

popt, pcov = opt.curve_fit(model, t, a, p0=[1e4, 1e-4, 1, 1e6, 3.5e-4, 1, 1e3, 5e-4, 1, 10], bounds=([-1e7, 0, 0, -1e7, 0, 0, -1e7, 0, 0, 0],[1e7, 1e-3, 6, 1e7, 1e-3, 6, 1e7, 1e-3, 6, 1e6]))

print(popt)
print(model(t, *popt))
fig = plt.figure()
#plt.plot(t, model(t, *popt) + 2*np.mean(a), "r--")
plt.plot(t, model(t, 1e6, 1e-4, 0, 1e6, 3.5e-4, 0, 1e6, 5e-4, 0, 0)+2*np.mean(a))
plt.plot(t, a, "k-")
plt.show()