import numpy as np
from mayavi import mlab

def test_surf():
    """Test surf on regularly spaced co-ordinates like MayaVi."""
    def f(x, y):
        sin, cos = np.sin, np.cos
        return sin(x + y) + sin(2 * x - y) + cos(3 * x + 4 * y)

    x, y = np.mgrid[-7.:7.05:0.1, -5.:5.05:0.05]
    s = mlab.surf(x, y, f)
    #cs = contour_surf(x, y, f, contour_z=0)
    return s

class Sphere():
    def __init__(self,x = 0, y = 0, z = 0, d = 10,c = (0,0,0), above_surface = True):
        d = d / 2
        if above_surface:
            shift = d
        else:
            shift = 0
        self.u = np.linspace(0, 2 * np.pi, 100)
        self.v = np.linspace(0, np.pi, 100)
        self.x = d * np.outer(np.cos(self.u), np.sin(self.v)) + x
        self.y = d * np.outer(np.sin(self.u), np.sin(self.v)) + y
        self.z = d * np.outer(np.ones(np.size(self.u)), np.cos(self.v)) + z + shift
        self.c = c
    def plot(self):
        mlab.mesh(self.x,self.y,self.z,color = self.c)
        
class RectangularPrism():
    def __init__(self,x = 0, y = 0, z= 0, w = 4, l = 4, h = 4,c = (0,0,0), above_surface = True):
        self.w = w /2
        self.l = l / 2
        self.h = h / 2     
        if above_surface:
            shift = self.h
        else:
            shift = 0
        r = [-1,1]
        self.X, self.Y = np.meshgrid(r, r)
        self.x = x
        self.z = z + shift
        self.y = y
        
        self.ones = np.ones((2,2))
        theta = np.pi / 3
        x = self.X
        y = self.Y
        #self.X = x * np.round(np.cos(theta),4) - y * np.round(np.sin(theta),4)
        #self.Y = y * np.round(np.cos(theta),4) - x * np.round(np.sin(theta),4)
        self.c = c
    def plot(self, c = (0,0,0)):
        def mesh(x,y,z,color):
            theta = np.pi /2
            theta = 0
            theta = np.pi
            
           # x = X * np.round(np.cos(theta),4) - Y * np.round(np.sin(theta),4)
            #y = Y * np.round(np.cos(theta),4) - X * np.round(np.sin(theta),4)
            mlab.mesh(x,y,z,color = color)
        mesh(self.X * self.w + self.x, self.Y * self.l + self.y, self.ones * self.h + self.z , color = self.c)
        mesh(self.X * self.w + self.x, self.Y * self.l + self.y,self.ones * -self.h + self.z,color = self.c)
        mesh(self.X * self.w + self.x, -self.l * self.ones + self.y,self.Y * self.h + self.z,color = self.c)
        mesh(self.X * self.w + self.x, self.ones*self.l + self.y,self.Y * self.h + self.z,color = self.c)
        mesh(self.ones * self.w + self.x,self.X * self.l + self.y,self.Y * self.h + self.z,color = self.c)
        mesh(-self.w * self.ones + self.x ,self.X * self.l + self.y,self.Y*self.h + self.z,color = self.c)
mlab.figure(size=(400, 300), bgcolor = (1,1,1))

#floor_3_3 = RectangularPrism(0,0,-1,w=23.5*3,l=23.5*3,h=2,c=(.39,.39,.39), above_surface = False)
d = 14.5 / np.sqrt(2)
m1 = RectangularPrism(-d,d,0,w=2,l=2,h=2,c = (1,1,0))
#m2 = Sphere(0,0,0,d = 2.75,c = (1,1,1))
m3 = Sphere(d,-d,0,d = 2.75,c = (1,1,1))

#floor_3_3.plot()
m1.plot()
#m2.plot()
m3.plot()

