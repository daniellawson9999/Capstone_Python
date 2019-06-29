# Author: Prabhu Ramachandran <prabhu [at] aero.iitb.ac.in>
# Copyright (c) 2009, Enthought, Inc.
# License: BSD Style.

from mayavi import mlab
from tvtk.tools import visual
# Create a figure
f = mlab.figure(size=(500,500))
# Tell visual to use this as the viewer.
visual.set_viewer(f)

# A silly visualization.
mlab.test_plot3d()

# Even sillier animation.
b1 = visual.box()

b2 = visual.box(x=4., color=(0,0,1))
b3 = visual.box(x=-4, color=(1,1,0))
b1.v = 5.0
b1._pos_changed()

@mlab.show
@mlab.animate(delay=250)
def anim():
    """Animate the b1 box."""
    while 1:
        b1.x = b1.x + b1.v*0.1
        if b1.x > 2.5 or b1.x < -2.5:
            b1.v = -b1.v
        yield

# Run the animation.
anim()