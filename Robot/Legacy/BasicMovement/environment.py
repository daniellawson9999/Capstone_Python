from mayavi import mlab
import numpy as np
from tvtk.tools import visual
import tvtk.tools
import matplotlib.pyplot as plt
from PIL import Image
from enum import Enum,auto

class State(Enum):
    WIN = auto()
    LOSS = auto()
    ILLEGAL = auto()
    STANDARD = auto()   

class Environment():
    win_reward = 100
    loss_reward = -100
    move_reward = -1
    def __init__(self):
        mlab.close(all=True)
        self.width = 900
        self.height = 500
        self.f = mlab.figure(size=(self.width,self.height),bgcolor = (1,1,1))
        self.f.scene._lift()
        visual.set_viewer(self.f)  
        self.square_width = 23.5
        a_side = 34.5
        tape_height = .2
        #distance between minerals
        d = 14.5 / np.sqrt(2)
        #color for silver
        silver = (.8,.8,.8)
        #reate field
        self.floor_3_3 = visual.box(x=0,y=0,z=-1, length = 23.5*3,height = 23.5*3,width = 2,color = (.4,.4,.4))  
        #randomize starting locations
        locations = [[-d,d],[0,0],[d,-d]]
        #np.random.shuffle(locations)
        #place minerals

        #self.gold_mineral = visual.box(x=locations[0][0],y=locations[0][1],z=1, length=4,height=4,width=4, color = (1,1,0))
        mineral_radius = 2.75 * 2
        self.gold_mineral = visual.sphere(x=locations[0][0],y=locations[0][1],z=mineral_radius,radius =mineral_radius,color = (1,1,0) )
        self.silver_mineral_1 = visual.sphere(x=locations[1][0],y=locations[1][1],z=mineral_radius,radius =mineral_radius,color = silver)
        self.silver_mineral_2 = visual.sphere(x=locations[2][0],y=locations[2][1],z=mineral_radius,radius =mineral_radius,color = silver)

        #randomly pick the red or blue side
        r = np.round(np.random.random(1)[0])
        b = 1 - r
        tape_color = (r,0,b)
        #23.5 is the diameter of a square
        #place the crater tape
        self.vertical_lander_tape = visual.box(x=-self.square_width*3/2 + 1,y=a_side/2 - self.square_width*3/2,z=tape_height,length = 2, height = a_side, width = tape_height,color=tape_color)
        self.h_lander_tape = visual.box(x=-self.square_width*3/2 + a_side/2,y=-a_side/2,z=tape_height,length = 2, height = a_side * np.sqrt(2), width = tape_height,color=tape_color)
        self.h_lander_tape.rotate(45,axis = [0,0,1],origin = [self.h_lander_tape.x,self.h_lander_tape.y,self.h_lander_tape.z])
        self.marker_left = visual.box(x=self.square_width/2 + 1, y =self.square_width,z=tape_height,length=2,height = self.square_width,width=tape_height,color=tape_color)
        self.marker_right = visual.box(x=3*self.square_width/2 -1, y =self.square_width,z=tape_height,length=2,height = self.square_width,width=tape_height,color=tape_color)
        self.marker_bottom = visual.box(x=self.square_width,y=self.square_width/2 + 1, z = tape_height,length=self.square_width,height=2,width=tape_height,color=tape_color)
        self.marker_top = visual.box(x=self.square_width,y=3*self.square_width/2 - 1, z = tape_height,length=self.square_width,height=2,width=tape_height,color=tape_color)

        #mlab.view(focalpoint=[d,d,0],distance=64, elevation=-80)
        self.x = -(self.square_width - 5) * np.random.random() - 5
        self.y = -(self.square_width - 5) * np.random.random() - 5
        self.update_position()
        
        self.move_distance = 2.5
    def update_position(self):
        angle = 0
        angle_r = angle * np.pi / 180
        view_distance = self.square_width * np.sqrt(2)
        shift = view_distance / np.sqrt(2)
        view_radius = view_distance / np.cos(angle_r)  
        #maybe add a z
        fp = [self.x+shift,self.y+shift,4]
        #print(fp)
        mlab.view(focalpoint=fp, distance=view_radius, elevation=-90 + angle, azimuth=45) 
        mlab.show()
    def move_position(self,left = 0,right = 0,forwards = 0,backwards = 0, set_value = True):
        #left first        
        x = self.x + (-left / np.sqrt(2)) + (right / np.sqrt(2)) + (forwards / np.sqrt(2) )- (backwards / np.sqrt(2))
        y = self.y + left / np.sqrt(2) - right / np.sqrt(2) + forwards / np.sqrt(2)  - backwards / np.sqrt(2)
        
        if set_value:
            self.x = x
            self.y = y
        else:
            return x,y
        
    def action_space(self):
        return 4
    
    def check_collision(self, mineral, x = None, y = None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if hasattr(mineral,'length'):
            rad = mineral.length / 2
        else:
            rad = mineral.radius
        if x <= mineral.x + rad and x >= mineral.x - rad and y <= mineral.y + rad and y >= mineral.y - rad:
            return True
        return False
    #returns an array of legal actions (ie [1,1,1,0])
    #action order: left, right, forwards, backwards       
      # state if robot is at given x,y. Defaults to the current x,y 
    def state(self,x = None,y = None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        apothem = self.square_width * 3 / 2
        if x > apothem or x < -apothem or y > apothem or y < -apothem:
            return State.ILLEGAL
        if self.check_collision(self.silver_mineral_1) or self.check_collision(self.silver_mineral_2):
            return State.LOSS
        if self.check_collision(self.gold_mineral):
            return State.WIN
        return State.STANDARD 
    
    def legal_actions(self):
        actions = [0,0,0,0]
        if self.state(*self.move_position(left = self.move_distance, set_value = False)) != State.ILLEGAL:
            actions[0] = 1
        if self.state(*self.move_position(right = self.move_distance, set_value = False)) != State.ILLEGAL:
            actions[1] = 1
        if self.state(*self.move_position(forwards = self.move_distance, set_value = False)) != State.ILLEGAL:
            actions[2] = 1
        if self.state(*self.move_position(backwards = self.move_distance, set_value = False)) != State.ILLEGAL:
            actions[3] = 1
        return actions
    
    def sample(self):
        actions = self.legal_actions()
        assert 1 in actions, "no legal actions to sample"
        action = np.random.randint(4)
        while actions[action] == 0:
            action = np.random.randint(4)
        return action
  
    #visual.box(x=33,y=22,z=1, length=2,height=2,width=2, color = (1,1,0))
    #checks the collision with a mineral at a given x,y. Defaults to the robot x,y
    def step(self,action):
        #action is a number 0-3
        assert (action >= 0 and action <= self.action_space() - 1), "action not in action space"
        #verify action is 
        assert (self.legal_actions()[action] == 1), "action not legal"
        
        moves = [0,0,0,0]
        moves[action] = self.move_distance
        
        #transition to new state
   
        self.move_position(*moves)
        self.update_position()
        
        #get the reward
        game_state = self.state()
        assert (game_state != State.ILLEGAL), "transitioned to an illegal state with action {} and distance".format(action,self.move_distance)
        
    
        if game_state == State.WIN:
            reward = self.win_reward
            done = True
        elif game_state == State.LOSS:
            reward = self.loss_reward
            done = True
        else:
            '''def distance(x1,y1,x2,y2):
                return np.sqrt((x2-x1)**2 + (y2-y1)**2)
            previous_distance = distance(previous_x,previous_y,self.gold_mineral.x,self.gold_mineral.y)
            current_distance = distance(self.x,self.y,self.gold_mineral.x,self.gold_mineral.y)
            if previous_distance > current_distance:
                reward = -1
            else:
                reward = -2'''
            reward = self.move_reward
            done = False
            
        next_state = self.screenshot()
        
        return next_state, reward, done, game_state
    def sample_image(self): 
        shot = mlab.screenshot()
        img = Image.fromarray(shot)
        #gray = img.convert('L')
        scale = 20
        resized = img.resize((round(np.shape(img)[1] / scale), round(np.shape(img)[0] / scale)), Image.ANTIALIAS)
        return resized
        #resized.save('test{}.png'.format(n))
    def screenshot(self):
        resized = self.sample_image()
        array = list(resized.getdata())
        #np.shape(img) for dimensions
        return list(np.asarray(array) / 255)
 
        
    def reset(self, random = True):
        #shuffle minerals
        d = 14.5 / np.sqrt(2)
        locations = [[-d,d],[0,0],[d,-d]]
        #np.random.shuffle(locations)
        #print(locations)
        self.gold_mineral.x = locations[0][0]
        self.gold_mineral.y = locations[0][1]
        self.silver_mineral_1.x = locations[1][0]
        self.silver_mineral_1.y = locations[1][1]
        self.silver_mineral_2.x = locations[2][0]
        self.silver_mineral_2.y = locations[2][1]
        
        if random:
            self.x = -(self.square_width - 5) * np.random.random() - 5
            self.y = -(self.square_width - 5) * np.random.random() - 5
        else: 
            self.x = -self.square_width
            self.y = -self.square_width
      

        r = np.round(np.random.random(1)[0])
        b = 1 - r
        tape_color = (r,0,b)
        
        self.vertical_lander_tape.color = tape_color
        self.h_lander_tape.color = tape_color
        self.marker_left.color = tape_color
        self.marker_right.color = tape_color
        self.marker_bottom.color = tape_color
        self.marker_top.color = tape_color
        self.update_position()
        return self.screenshot()
        

    
    
        
        #also update side color and mineral positions
    
    
        