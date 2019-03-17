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

class Location(Enum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2

class Reward(Enum):
    TERMINAL = auto()
    RELATIVE = auto()
    PROPORTIONAL = auto()

class Action(Enum):
    LEFT = auto()
    RIGHT = auto()
    FORWARDS = auto()
    BACKWARDS = auto()
    CW = auto()
    CCW = auto()
    
class Environment():  
    square_width = 23.5
    win_reward = 100
    loss_reward = -100
    move_reward = -1
    def get_pos_angle(self,x,y):
        angle = np.arctan(y/x) * 180 / np.pi
        if x < 0:
            angle += 180
        return angle
    def random_position(self):
            
        def rand_coordinate():
            return (np.random.random() * Environment.square_width  + Environment.square_width / 2) * np.power(-1,np.random.randint (0,2))
        x = rand_coordinate()
        y = rand_coordinate()
        angle = self.get_pos_angle(x,y)
        return x,y,angle
    
    def get_start_position(self):
        if self.random_location:
            return self.random_position()
        else:
            #the fixed starting position
            x = -self.square_width
            y = -self.square_width
            angle = self.get_pos_angle(x,y)
            return x,y,angle
            
            
    def get_mineral_locations(self):
        d = self.d
        if self.mineral_location == Location.LEFT:
            locations = [[-d,d],[0,0],[d,-d]]
        elif(self.mineral_location == Location.RIGHT):
            locations = [[d,-d],[-d,d],[0,0]]
        else:
            locations = [[0,0],[-d,d],[d,-d]]        
        if self.random_minerals:
            np.random.shuffle(locations)
        return locations
    
    def get_action_index_dictionary(self):
        d = {}
        #initialize each action as None by iterating through the enum Action and setting in a dictionary
        for a in Action:
            d[a] = None
        #assign an index to each available action, leave actions not being used as null
        index = 0
        for a in self.actions:
            d[a] = index
            index += 1
        return d
    
    def __init__(self,random_minerals=True,random_location=True, mineral_location= Location.CENTER,
                 reward = Reward.RELATIVE, grayscale = False, flat = False,
                 actions = [Action.LEFT,Action.RIGHT,Action.FORWARDS,Action.BACKWARDS,Action.CW,Action.CCW]):
        
        self.random_minerals = random_minerals
        self.random_location = random_location
        self.mineral_location = mineral_location
        self.reward = reward
        self.grayscale = grayscale
        self.flat = flat
        self.actions = actions
        self.actions_index_dict = self.get_action_index_dictionary()
        mlab.close(all=True)
        self.width = 900
        self.height = 500
        self.f = mlab.figure(size=(self.width,self.height),bgcolor = (1,1,1))
        self.f.scene._lift()
        self.square_width = 23.5
        visual.set_viewer(self.f)  
        a_side = 34.5
        tape_height = .2
        #distance between minerals
        self.d = 14.5 / np.sqrt(2)
        #color for silver
        #silver = (.8,.8,.8)
        silver = (.5,.5,.7)
        #reate field
        self.floor_3_3 = visual.box(x=0,y=0,z=-1, length = 23.5*3,height = 23.5*3,width = 2,color = (.4,.4,.4))  
        #get mineral location
        locations = self.get_mineral_locations()
        #self.gold_mineral = visual.box(x=locations[0][0],y=locations[0][1],z=1, length=4,height=4,width=4, color = (1,1,0))
        mineral_radius = 2.75 * 1.5
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

        self.x, self.y, self.pos_angle = self.get_start_position()
        self.init_position()
        self.move_distance = 2
        self.turn_angle = 5
    def init_position(self):
        angle = 0
        angle_r = angle * np.pi / 180
        view_distance = self.square_width * np.sqrt(2)
        shift = view_distance / np.sqrt(2)
        view_radius = view_distance / np.cos(angle_r)  
        #maybe add a z
        rad  = np.deg2rad(self.pos_angle)
        #fp = [self.x-shift*np.sign(np.cos(rad)),self.y-shift*np.sign(np.sin(rad)),0]
        fp = [self.x-shift*np.cos(rad),self.y-shift*np.sin(rad),4]
        #print(fp)
        mlab.view(focalpoint=fp, distance=view_radius, elevation=-90 + angle, azimuth=self.pos_angle)
        mlab.show()
    
    def move_position(self,left = 0,right = 0,forwards = 0,backwards = 0, pos_angle = 0, neg_angle = 0, set_value = True):
        mlab.move(forwards - backwards, right - left,0)
        mlab.yaw(pos_angle - neg_angle)
        
    def action_space(self):
        return len(self.actions)
    
    def check_collision(self, mineral, x = None, y = None):
        pos = mlab.move()[0]
        if x is None:
            x = pos[0]
        if y is None:
            y = pos[1]
        if hasattr(mineral,'length'):
            rad = mineral.length / 2
        else:
            rad = mineral.radius
        rad *= 1.2
        if x <= mineral.x + rad and x >= mineral.x - rad and y <= mineral.y + rad and y >= mineral.y - rad:
            return True
        return False
    #returns an array of legal actions (ie [1,1,1,0])
    #action order: left, right, forwards, backwards       
      # state if robot is at given x,y. Defaults to the current x,y 
    def state(self,x = None,y = None):
        pos = mlab.move()[0]
        if x is None:
            x = pos[0]
        if y is None:
            y = pos[1]
        apothem = self.square_width * 3 / 2
        if x > apothem or x < -apothem or y > apothem or y < -apothem:
            return State.ILLEGAL
        if self.check_collision(self.silver_mineral_1) or self.check_collision(self.silver_mineral_2):
            return State.LOSS
        if self.check_collision(self.gold_mineral):
            return State.WIN
        return State.STANDARD 
    
    def legal_actions(self):
        actions = [0,0,0,0,1,1]
        actions = [0] * self.action_space()
        pos,focal = mlab.move()
        v = focal - pos
        v = np.asarray((v[0],v[1]))
        v /= np.linalg.norm(v)
        w = np.asarray((-v[1],v[0]))
        pos = np.asarray((pos[0],pos[1]))
        #left
        if self.state(*(self.move_distance * w + pos)) != State.ILLEGAL:
            actions[0] = 1
        #right
        if self.state(*(self.move_distance * -w + pos)) != State.ILLEGAL:
            actions[1] = 1
        #forwards
        if self.state(*(self.move_distance * v + pos)) != State.ILLEGAL:
            actions[2] = 1
        #backwards
        if self.state(*(self.move_distance * -v + pos)) != State.ILLEGAL:
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
        
        moves = [0] * self.action_space()
        if action == 4 or action == 5:
            moves[action] = self.turn_angle
        else:
            moves[action] = self.move_distance
        #store previous position
        previous_pos = mlab.move()[0]
        #transition to new state
        self.move_position(*moves)
        new_pos = mlab.move()[0]
        
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
            done = False
            def distance(x1,y1,x2,y2):
                return np.sqrt((x2-x1)**2 + (y2-y1)**2)
            previous_distance = distance(previous_pos[0],previous_pos[1],self.gold_mineral.x,self.gold_mineral.y)
            current_distance = distance(new_pos[0],new_pos[1],self.gold_mineral.x,self.gold_mineral.y)
            if self.reward == Reward.RELATIVE:
                if previous_distance > current_distance:
                    reward = self.move_reward
                else:
                    reward = self.move_reward * 2
            elif self.reward == Reward.PROPORTIONAL:
                reward = current_distance**2 / -100
            else:
                reward = self.move_reward
                    #reward = self.move_reward
               
            
        next_state = self.screenshot()
        
        return next_state, reward, done, game_state
    def sample_image(self): 
        shot = mlab.screenshot()
        img = Image.fromarray(shot)
        if self.grayscale:
            img = img.convert('L')
        scale = 15
        resized = img.resize((round(np.shape(img)[1] / scale), round(np.shape(img)[0] / scale)), Image.ANTIALIAS)
        return resized
        #resized.save('test{}.png'.format(n))
    def screenshot(self):
        resized = self.sample_image()
        if self.flat:
            array = resized.getdata()
            array = np.asarray(array)
            array = array.ravel()
        else:
            array = np.asarray(resized)
        array = array / 255
        #array = np.asarray(resized)
        #np.shape(img) for dimensions
        return array
 
        
    def reset(self):
        #shuffle minerals
        locations = self.get_mineral_locations()
        #print(locations)
        self.gold_mineral.x = locations[0][0]
        self.gold_mineral.y = locations[0][1]
        self.silver_mineral_1.x = locations[1][0]
        self.silver_mineral_1.y = locations[1][1]
        self.silver_mineral_2.x = locations[2][0]
        self.silver_mineral_2.y = locations[2][1]
        
        self.x, self.y, self.pos_angle = self.get_start_position()
        
        r = np.round(np.random.random(1)[0])
        b = 1 - r
        tape_color = (r,0,b)
        
        self.vertical_lander_tape.color = tape_color
        self.h_lander_tape.color = tape_color
        self.marker_left.color = tape_color
        self.marker_right.color = tape_color
        self.marker_bottom.color = tape_color
        self.marker_top.color = tape_color
        self.init_position()
        return self.screenshot()
        

    
    
        
        #also update side color and mineral positions
    
    
        