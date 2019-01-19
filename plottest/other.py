import mayavi
from mayavi import mlab
import numpy as np
from tvtk.tools import visual
import matplotlib.pyplot as plt
from PIL import Image
import time
class Environment():        
    def __init__(self):
        mlab.close(all=True)
        f = mlab.figure(size=(900,500),bgcolor = (1,1,1))
        visual.set_viewer(f)  
        square_width = 23.5
        self.square_width = 23.5
        a_side = 34.5
        tape_height = .2
        #mlab.view(focalpoint=[0,0,0],elevation=180,figure=f)
        #mlab.roll(0)
        #distance between minerals
        d = 14.5 / np.sqrt(2)
        #color for silver
        silver = (.8,.8,.8)
        #reate field
        floor_3_3 = visual.box(x=0,y=0,z=-1, length = 23.5*3,height = 23.5*3,width = 2,color = (.4,.4,.4))  
        #randomize starting locations
        locations = [[-d,d],[0,0],[d,-d]]
        np.random.shuffle(locations)
        #place minerals
        gold_mineral = visual.box(x=locations[0][0],y=locations[0][1],z=1, length=2,height=2,width=2, color = (1,1,0))
        silver_mineral_1 = visual.sphere(x=locations[1][0],y=locations[1][1],z=2.75/2,radius =2.75/2,color = silver)
        silver_mineral_2 = visual.sphere(x=locations[2][0],y=locations[2][1],z=2.75/2,radius =2.75/2,color = silver)
    
        #randomly pick the red or blue side
        r = np.round(np.random.random(1)[0])
        b = 1 - r
        tape_color = (r,0,b)
        #23.5 is the diameter of a square
        #place the crater tape
        vertical_lander_tape = visual.box(x=-square_width*3/2 + 1,y=a_side/2 - square_width*3/2,z=tape_height,length = 2, height = a_side, width = tape_height,color=tape_color)
        h_lander_tape = visual.box(x=-square_width*3/2 + a_side/2,y=-a_side/2,z=tape_height,length = 2, height = a_side * np.sqrt(2), width = tape_height,color=tape_color)
        h_lander_tape.rotate(45,axis = [0,0,1],origin = [h_lander_tape.x,h_lander_tape.y,h_lander_tape.z])
        marker_left = visual.box(x=square_width/2 + 1, y =square_width,z=tape_height,length=2,height = square_width,width=tape_height,color=tape_color)
        marker_right = visual.box(x=3*square_width/2 -1, y =square_width,z=tape_height,length=2,height = square_width,width=tape_height,color=tape_color)
        marker_bottom = visual.box(x=square_width,y=square_width/2 + 1, z = tape_height,length=square_width,height=2,width=tape_height,color=tape_color)
        marker_bottom = visual.box(x=square_width,y=3*square_width/2 - 1, z = tape_height,length=square_width,height=2,width=tape_height,color=tape_color)

        #mlab.view(focalpoint=[d,d,0],distance=64, elevation=-80)
        self.x = -square_width
        self.y = -square_width
        self.update_position()
        

        
    def update_position(self):
        angle_d = 10
        angle_r = 10 * np.pi / 180
        view_distance = self.square_width * np.sqrt(2)
        shift = view_distance / np.sqrt(2)
        view_radius = view_distance / np.cos(angle_r)  
        #maybe add a z
        fp = [self.x+shift,self.y+shift,0]
        #print(fp)
        mlab.view(focalpoint=fp, distance=view_radius, elevation=-90 + angle_d, azimuth=45) 
        mlab.show()
    def move_position(self,left = 0,right = 0,forwards = 0,backwards = 0):
        #left first
        self.x += (-left / np.sqrt(2)) + (right / np.sqrt(2)) + (forwards / np.sqrt(2) )- (backwards / np.sqrt(2))
        self.y += left / np.sqrt(2) - right / np.sqrt(2) + forwards / np.sqrt(2)  - backwards / np.sqrt(2)
env = Environment()

@mlab.animate(delay = 2000)
def anim1():
    env.move_position(left= 10)
    env.update_position()
    yield
    
@mlab.animate()
def anim2(delay = 2000):
    '''
    for i in range(20):
        env.move_position(forwards = 1)
        env.update_position()
        #global img
        #img = mlab.screenshot()
        #mlab.savefig('test.png')
        print(i)
        yield
    for j in range(20):
        env.move_position(right = 1)
        env.update_position()
        global img
        #img = mlab.screenshot()
        #mlab.savefig('test.png')
        #print(i)
        yield
    '''
    def save_image(n):
        img = mlab.screenshot()
        i = Image.fromarray(img)
        gray = i.convert('L')
        scale = 10
        resized = gray.resize((round(np.shape(img)[1] / scale), round(np.shape(img)[0] / scale)), Image.ANTIALIAS)
        resized.save('test{}.png'.format(n))
    save_image(0)
    env.move_position(forwards =10)
    env.update_position()
    save_image(1)
    yield
    env.move_position(right= 10)
    env.update_position()
    save_image(2)
    yield
    env.move_position(left= 10)
    env.update_position()
    save_image(3)
#anim1()
anim2()
#my_img = np.array()
#mlab.close()
