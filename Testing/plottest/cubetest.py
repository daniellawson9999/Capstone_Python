from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
w = 5
l = 3
h = 10
r = [-1,1]
X, Y = np.meshgrid(r, r)
x = 0
y = 0 
z = 0
ones = np.ones((2,2))
ax.plot_surface(X * w + x, Y * l + y, ones * h + z , color = 'b')
ax.plot_surface(X * w + x, Y * l + y,ones * -h + z,color = 'b')
ax.plot_surface(X * w + x, -l * ones + y,Y * h + z,color = 'b')
ax.plot_surface(X * w + x, ones*l + y,Y * h + z,color = 'b')
ax.plot_surface(ones * w + x,X * l + y,Y * h + z,color = 'b')
ax.plot_surface(-w * ones + x ,X * l + y,Y*h + z,color = 'b')

