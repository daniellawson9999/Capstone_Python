from mayavi import mlab
import numpy as np
from tvtk.tools import visual
import tvtk.tools
import matplotlib.pyplot as plt
from PIL import Image
from enum import Enum,auto
import copy
import cv2



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
    RELATIVE_PROPORTIONAL = auto()

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
            return ( (np.random.random() * (Environment.square_width + 2))  + (Environment.square_width) / 2 -2) * np.power(-1,np.random.randint (0,2))
        x = rand_coordinate()
        y = rand_coordinate()
        angle = self.get_pos_angle(x,y)
        return x,y,angle
    
    def get_start_position(self):
        if self.random_location:
            return self.random_position()
        else:
            #the fixed starting position
            x = self.start_pos - self.start_shift
            y = x
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
                 mineral_scale = 1.5, start_shift = 0, camera_height = 4,
                 actions = [Action.LEFT,Action.RIGHT,Action.FORWARDS,Action.BACKWARDS,Action.CW,Action.CCW],
                 decorations = False, camera_tilt =  0,start_pos=-23.5,
                 width = 900, height = (500-46),resize_scale=15,
                 x_collision_scale = 1,y_collision_scale = 1,k=5,silver=(.5,.5,.7), random_colors = False,random_lighting=False):
        
        self.random_minerals = random_minerals
        self.random_location = random_location
        self.mineral_location = mineral_location
        self.reward = reward
        self.grayscale = grayscale
        self.flat = flat
        self.actions = actions.copy()
        self.actions_index_dict = self.get_action_index_dictionary()
        self.camera_height = camera_height
        self.decorations = decorations
        self.camera_tilt = camera_tilt
        self.start_pos = start_pos
        self.resize_scale = resize_scale
        mlab.close(all=True)
        self.width = width
        self.height = height + 46
        self.f = mlab.figure(size=(self.width,self.height),bgcolor = (1,1,1))
        
        self.f.scene._lift()
        self.square_width = 23.5
        self.start_shift = start_shift
        self.x_collision_scale = x_collision_scale
        self.y_collision_scale = y_collision_scale
        self.k = k
        self.k_max_iterations = 10
        self.silver = silver
        self.random_colors = random_colors
        self.random_lighting = random_lighting
        visual.set_viewer(self.f)  
        a_side = 34.5
        tape_height = .2
        #distance between minerals
        self.d = 14.5 / np.sqrt(2)
        #color for silver
        #silver = (.8,.8,.8)
        floor_color = (.4,.4,.4)
        #reate field
        self.floor_3_3 = visual.box(x=0,y=0,z=-1, length = 23.5*3,height = 23.5*3,width = 2,color = floor_color)  
        #get mineral location
        locations = self.get_mineral_locations()
        #self.gold_mineral = visual.box(x=locations[0][0],y=locations[0][1],z=1, length=4,height=4,width=4, color = (1,1,0))
        mineral_radius = 2.75 * mineral_scale
        self.gold_mineral = visual.sphere(x=locations[0][0],y=locations[0][1],z=mineral_radius,radius =mineral_radius,color = (1,1,0) )
        self.silver_mineral_1 = visual.sphere(x=locations[1][0],y=locations[1][1],z=mineral_radius,radius =mineral_radius,color = self.silver)
        self.silver_mineral_2 = visual.sphere(x=locations[2][0],y=locations[2][1],z=mineral_radius,radius =mineral_radius,color = self.silver)

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

        #add bars
        if self.decorations:
            bar_width = 1.5
            bar_height = 1
            middle_height = 12 - bar_height*2
            middle_color = floor_color
            bar_length = self.square_width * 3
            bar_color = (0,0,0)
            self.bar1 = visual.box(x=self.square_width*1.5-bar_width/2,y=0,z=tape_height, width= bar_width, height=bar_height,length=bar_length, color = bar_color)
            self.bar1.rotate(90,axis=[0,0,1],origin=self.bar1.pos)
            
            self.bar1m = visual.box(x=self.square_width*1.5-bar_width/2,y=0,z=bar_height+middle_height/2, width= middle_height, height=bar_width,length=bar_length, color = middle_color)
            self.bar1m.rotate(90,axis=[0,0,1],origin=self.bar1m.pos)
            
            self.bar1t = visual.box(x=self.square_width*1.5-bar_width/2,y=0,z=bar_height+middle_height, width= bar_height, height=bar_width,length=bar_length, color = bar_color)
            self.bar1t.rotate(90,axis=[0,0,1],origin=self.bar1t.pos)

            
            self.bar2 = visual.box(x=-self.square_width*1.5+bar_width/2,y=0,z=tape_height, width= bar_width, height=bar_height,length=bar_length, color = bar_color)
            self.bar2.rotate(90,axis=[0,0,1],origin=self.bar2.pos)
            
            self.bar2m = visual.box(x=-self.square_width*1.5+bar_width/2,y=0,z=bar_height+middle_height/2, width= middle_height, height=bar_width,length=bar_length, color = middle_color)
            self.bar2m.rotate(90,axis=[0,0,1],origin=self.bar2m.pos)
            
            self.bar2t = visual.box(x=-self.square_width*1.5+bar_width/2,y=0,z=bar_height+middle_height, width= bar_height, height=bar_width,length=bar_length, color = bar_color)
            self.bar2t.rotate(90,axis=[0,0,1],origin=self.bar2t.pos)
            
            
            
            self.bar3 = visual.box(x=0,y=self.square_width*1.5-bar_width/2,z=tape_height, width= bar_width, height=bar_height,length=bar_length, color = bar_color)
            
            self.bar3m = visual.box(x=0,y=self.square_width*1.5-bar_width/2,z=bar_height+middle_height/2, width= middle_height, height=bar_width,length=bar_length, color = middle_color)

            self.bar3t = visual.box(x=0,y=self.square_width*1.5-bar_width/2,z=bar_height+middle_height, width= bar_height, height=bar_width,length=bar_length, color = bar_color)

            
            self.bar4 = visual.box(x=0,y=-self.square_width*1.5+bar_width/2,z=tape_height, width= bar_width, height=bar_height,length=bar_length, color = bar_color)
            
            self.bar4m = visual.box(x=0,y=-self.square_width*1.5+bar_width/2,z=bar_height+middle_height/2, width= middle_height, height=bar_width,length=bar_length, color = middle_color)

            self.bar4t = visual.box(x=0,y=-self.square_width*1.5+bar_width/2,z=bar_height+middle_height, width= bar_height, height=bar_width,length=bar_length, color = bar_color)
            
            if self.random_colors:
                height_scale = 40
                new_height =  bar_height * height_scale
                self.bar1t.width= new_height
                self.bar1t.rotate(90,axis=[0,0,1],origin=self.bar1t.pos)
                self.bar2t.width= new_height
                self.bar2t.rotate(90,axis=[0,0,1],origin=self.bar2t.pos)
                self.bar3t.width = new_height
                self.bar4t.width = new_height
                
                self.randomize_colors()
                
                
            
        self.x, self.y, self.pos_angle = self.get_start_position()
        self.init_position()
        if self.random_lighting:
            self.randomize_lighting()
        self.move_distance = 2
        self.turn_angle = 5
    #from colorsys library source code
    def hsv_to_rgb(self,h, s, v):
        if s == 0.0:
            return v, v, v
        i = int(h*6.0) # XXX assume int() truncates!
        f = (h*6.0) - i
        p = v*(1.0 - s)
        q = v*(1.0 - s*f)
        t = v*(1.0 - s*(1.0-f))
        i = i%6
        if i == 0:
            return v, t, p
        if i == 1:
            return q, v, p
        if i == 2:
            return p, v, t
        if i == 3:
            return p, q, v
        if i == 4:
            return t, p, v
        if i == 5:
            return v, p, q
        # Cannot get here
    def randomize_colors(self):
        r = np.round(np.random.random(1)[0])
        b = 1 - r
        tape_color = (r,0,b)
        if (self.random_colors):
            #change this to a randomly generated color
            min_value = 10
            random_value = (np.random.randint(100-min_value) + min_value) / 100
            min_saturation = 25
            random_saturation = (np.random.randint(100-min_saturation) + min_saturation) / 100
            hue_yellow = [30,75]
            random_hue = np.random.choice(np.concatenate((np.arange(hue_yellow[0]),np.arange(360-hue_yellow[1])+hue_yellow[1]+1))) / 360
            color = self.hsv_to_rgb(random_hue,random_saturation,random_value)
            objects = [self.floor_3_3,self.bar1,self.bar2,self.bar3,self.bar4,self.bar1m,self.bar2m,self.bar3m,self.bar4m,self.bar1t,self.bar2t,self.bar3t,self.bar4t]
            for o in objects:
                o.color = color
            if np.round(np.random.random(1)[0]):
                tape_color = color
        
        self.vertical_lander_tape.color = tape_color
        self.h_lander_tape.color = tape_color
        self.marker_left.color = tape_color
        self.marker_right.color = tape_color
        self.marker_bottom.color = tape_color
        self.marker_top.color = tape_color
    def randomize_lighting(self):
        for i in range(3):
            self.f.scene.light_manager.lights[i].intensity = np.random.rand() / 2 + .5
    def init_position(self):
        angle = self.camera_tilt
        angle_r = angle * np.pi / 180
        view_distance = self.square_width * np.sqrt(2)
        shift = view_distance / np.sqrt(2)
        view_radius = view_distance / np.cos(angle_r)  
        #maybe add a z
        rad  = np.deg2rad(self.pos_angle)
        #fp = [self.x-shift*np.sign(np.cos(rad)),self.y-shift*np.sign(np.sin(rad)),0]
        fp = [self.x-shift*np.cos(rad),self.y-shift*np.sin(rad),self.camera_height]
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
        if x <= mineral.x + rad * self.x_collision_scale and x >= mineral.x - rad * self.x_collision_scale  and y <= mineral.y + rad *self.y_collision_scale and y >= mineral.y - rad * self.y_collision_scale:
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
        actions = [0] * self.action_space()
        #abbreviate actions_index_dict to d
        d = self.actions_index_dict
        
        #Set turning to be legal if they are availabe actions
        if d[Action.CW] != None:
            actions[d[Action.CW]] = 1
        if d[Action.CCW] != None:
            actions[d[Action.CCW]] = 1
            
        pos,focal = mlab.move()
        v = focal - pos
        v = np.asarray((v[0],v[1]))
        v /= np.linalg.norm(v)
        w = np.asarray((-v[1],v[0]))
        pos = np.asarray((pos[0],pos[1]))
        
        #left
        if self.state(*(self.move_distance * w + pos)) != State.ILLEGAL and d[Action.LEFT] != None:
            actions[d[Action.LEFT]] = 1
        #right
        if self.state(*(self.move_distance * -w + pos)) != State.ILLEGAL and d[Action.RIGHT] != None:
            actions[d[Action.RIGHT]] = 1
        #forwards
        if self.state(*(self.move_distance * v + pos)) != State.ILLEGAL and d[Action.FORWARDS] != None:
            actions[d[Action.FORWARDS]] = 1
        #backwards
        if self.state(*(self.move_distance * -v + pos)) != State.ILLEGAL and d[Action.BACKWARDS] != None:
            actions[d[Action.BACKWARDS]] = 1
        return actions
    
    def sample(self):
        actions = self.legal_actions()
        assert 1 in actions, "no legal actions to sample"
        action = np.random.randint(self.action_space())
        while actions[action] == 0:
            action = np.random.randint(self.action_space())
        return action
    
    #visual.box(x=33,y=22,z=1, length=2,height=2,width=2, color = (1,1,0))
    #checks the collision with a mineral at a given x,y. Defaults to the robot x,y
    def step(self,action):
        
        #action is in the action space
        assert (action >= 0 and action <= self.action_space() - 1), "action not in action space"
        #verify action is legal
        assert (self.legal_actions()[action] == 1), "action not legal"

        #left = 0,right = 0,forwards = 0,backwards = 0, pos_angle = 0, neg_angle = 0
  
        #get the action which coressponds to the action integer
        action_name = None
        for act, index in self.actions_index_dict.items():
            if index == action:
                action_name = act
        #make sure that the action is found
        assert (action_name != None), "action not found"
        
        #create moves list to pass to the move_position function
        moves = [0] * 6
        #assign move or turn angle, use the enum value -1 because each enum value is ones based
        if action_name == Action.CW or action_name == Action.CCW:
            moves[action_name.value - 1] = self.turn_angle
        else:
            moves[action_name.value - 1] = self.move_distance
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
            #calculate previous and current distances
            def distance(x1,y1,x2,y2):
                return np.sqrt((x2-x1)**2 + (y2-y1)**2)
            previous_distance = distance(previous_pos[0],previous_pos[1],self.gold_mineral.x,self.gold_mineral.y)
            current_distance = distance(new_pos[0],new_pos[1],self.gold_mineral.x,self.gold_mineral.y)
            
            #use one of the reward structures
            if self.reward == Reward.RELATIVE:
                if previous_distance > current_distance:
                    reward = self.move_reward
                else:
                    reward = self.move_reward * 2
            elif self.reward == Reward.PROPORTIONAL:
                reward = current_distance**2 / -100
            elif self.reward == Reward.RELATIVE_PROPORTIONAL:
                reward = previous_distance - current_distance
                #if reward == 0:
                    #reward = -.5
            else:
                reward = self.move_reward
            #also end the game if there are no more legal actions in the new state 
            if max(self.legal_actions()) == 0:
               done = True
            
        next_state = self.screenshot()
        
        return next_state, reward, done, game_state
    def sample_image(self): 
        shot = mlab.screenshot()
        img = Image.fromarray(shot)
        if self.grayscale:
            img = img.convert('L')
        scale = self.resize_scale
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
        return array;
        
    #segmented image through screenshot_segmented implements k means clustering for image segmentation.
    def segmented_image(self, array):
        original_shape = array.shape
        array = array.reshape((-1,3))
        array = np.float32(array)
        
        #kmeans code follows the form of the opencv docs
        max_iterations = self.k_max_iterations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations,1.0)
        ret,label,center=cv2.kmeans(array,self.k,None,criteria,max_iterations,cv2.KMEANS_PP_CENTERS )
        
        center = np.uint8(center)
        res = center[label.flatten()]
        reshaped = res.reshape((original_shape))
        
        return reshaped
    
    
    def display_segmented_image(self,array):
        img = Image.fromarray(self.segmented_image(array))
        return img
    
    def display_resized_image(self, array):
        array  = self.segmented_image(array)
        img = Image.fromarray(array)
        scale = self.resize_scale
        resized_img = img.resize((round(np.shape(img)[1] / scale), round(np.shape(img)[0] / scale)), Image.ANTIALIAS)
        return resized_img
    
    def screenshot_segmented(self):
        img = self.display_resized_image(mlab.screenshot())
        array = np.asarray(img)
        array = array / 255
        return array
        
        
    
    
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
        
        self.randomize_colors()
        
        if self.random_lighting:
            self.randomize_lighting()
        
        self.init_position()
        return self.screenshot()    
