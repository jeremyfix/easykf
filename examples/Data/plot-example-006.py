from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
 
data = np.loadtxt("Output/example-006.data")

fig = plt.figure()
ax = Axes3D(fig, azim = -128, elev = 43)
s = .1
X = np.arange(-4, 4.+s, s) # -2 2
Y = np.arange(-15, 15.+s, s) # -1 3
X, Y = np.meshgrid(X, Y)
Z = (1.-X)**2 + 100.*(Y-X*X)**2
ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, norm = LogNorm(), cmap = cm.jet)
ax.plot(data[:,0], data[:,1], data[:,2], color='r')

plt.xlabel("x")
plt.ylabel("y")
 
 
plt.show()
