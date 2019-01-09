# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

class Sphere():
    def __init__(self,x = 0, y = 0, z = 0, d = 10):
        d = d / 2
        self.u = np.linspace(0, 2 * np.pi, 100)
        self.v = np.linspace(0, np.pi, 100)
        self.x = d * np.outer(np.cos(self.u), np.sin(self.v)) + x
        self.y = d * np.outer(np.sin(self.u), np.sin(self.v)) + y
        self.z = d * np.outer(np.ones(np.size(self.u)), np.cos(self.v)) + z
    def plot(self, c = 'r'):
        ax.plot_surface(self.x,self.y, self.z,color = c)

class RectangularPrism():
    def __init__(self,x = 0, y = 0, z= 0, w = 4, l = 4, h = 4):
        self.w = w /2
        self.l = l / 2
        self.h = h / 2
        r = [-1,1]
        self.X, self.Y = np.meshgrid(r, r)
        self.x = x
        self.y = y
        self.z = z
        self.ones = np.ones((2,2))
    def plot(self, c = 'r'):
        ax.plot_surface(self.X * self.w + self.x, self.Y * self.l + self.y, self.ones * self.h + self.z , color = c)
        ax.plot_surface(self.X * self.w + self.x, self.Y * self.l + self.y,self.ones * -self.h + self.z,color = c)
        ax.plot_surface(self.X * self.w + self.x, -self.l * self.ones + self.y,self.Y * self.h + self.z,color = c)
        ax.plot_surface(self.X * self.w + self.x, self.ones*self.l + self.y,self.Y * self.h + self.z,color = c)
        ax.plot_surface(self.ones * self.w + self.x,self.X * self.l + self.y,self.Y * self.h + self.z,color = c)
        ax.plot_surface(-self.w * self.ones + self.x ,self.X * self.l + self.y,self.Y*self.h + self.z,color = c)
        
floor = RectangularPrism(0,0,-2,10,10,2)
sphere1 = Sphere(3,1,0,2)
sphere2 = Sphere(3,0,0,2)
sphere3 = Sphere(3,-1,0,2)



sphere1.plot()
sphere2.plot()
sphere3.plot()
floor.plot(c= 'b')
ax.set_axis_off()
axisEqual3D(ax)
for i in range(360):
    ax.view_init(elev=5., azim=i)
    plt.draw()
    plt.pause(.001)
# Plot the surface
#ax.plot_surface(sphere1.x, sphere1.y, sphere1.z, color='b')
#ax.plot_surface(sphere2.x,sphere2.y, sphere2.z,color = 'r')
#ax.plot_surface(cube1.x, cube1.y, cube1.z)