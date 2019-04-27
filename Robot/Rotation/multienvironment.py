from mayavi import mlab
import numpy as np
from tvtk.tools import visual
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import tvtk.tools
import matplotlib.pyplot as plt
from PIL import Image
from enum import Enum,auto
import copy
import cv2
import random



class Mineral(Enum):
    GOLD = auto()
    SILVER = auto()
    
class State(Enum):
    WIN = auto()
    LOSS = auto()
    ILLEGAL = auto()
    STANDARD = auto()  
    GOLD_COLLISION = auto()
    SILVER_COLLISION = auto()
    WALL_COLLISION = auto()

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
    STAY = auto()
    
class Environment():  
    square_width = 23.5
    win_reward = 100
    loss_reward = -100
    move_reward = -1
            
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
    
    def __init__(self,
                 reward = Reward.RELATIVE, grayscale = False, flat = False,
                 mineral_scale = 1.5, camera_height = 4,
                 actions = [Action.LEFT,Action.RIGHT,Action.FORWARDS,Action.BACKWARDS,Action.CW,Action.CCW],
                 decorations = False, camera_tilt =  0,
                 width = 900, height = (500-46),resize_scale=15,
                 k=5,silver=(.5,.5,.7), random_colors = False,random_lighting=False,
                 silver_mineral_num = 3, point_distance = 9, stationary_scale =6, normal_scale = 2, stationary_win_count = 5):
        
        self.reward = reward
        self.grayscale = grayscale
        self.flat = flat
        self.actions = actions.copy()
        self.actions_index_dict = self.get_action_index_dictionary()
        self.camera_height = camera_height
        self.decorations = decorations
        self.camera_tilt = camera_tilt
        self.resize_scale = resize_scale

        self.square_width = 23.5
        self.k = k
        self.k_max_iterations = 10
        self.silver = silver
        self.random_colors = random_colors
        self.random_lighting = random_lighting
        self.silver_mineral_num = silver_mineral_num
        self.point_distance = point_distance
        self.stationary_scale = stationary_scale
        self.normal_scale = normal_scale
        self.stationary_win_count = stationary_win_count
        self.stationary_count = 0
        self.exclude_zone = -1
        self.mineral_scale = mineral_scale
    
        mlab.close(all=True)
        self.width = width
        self.height = height + 46
        self.f = mlab.figure(size=(self.width,self.height),bgcolor = (1,1,1))
        self.f.scene._lift()
       
        
        visual.set_viewer(self.f) 
        
        self.move_distance = 2
        self.turn_angle = 5
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
        #self.gold_mineral = visual.box(x=locations[0][0],y=locations[0][1],z=1, length=4,height=4,width=4, color = (1,1,0))
        self.mineral_radius = 2.75 * self.mineral_scale
        
        #the first zone is top left corner, then moves across the first row, and then to the next row
        self.zones = []
        start_y = self.square_width
        start_x = -self.square_width
        for i in range(3):
            for j in range(3):
                self.zones.append([start_x + self.square_width * j, start_y - self.square_width*i])
        self.zones = np.asarray(self.zones)
        self.minerals = []
        self.init_mineral_list()
        self.reset_minerals()
        #self.gold_mineral = visual.sphere(x=locations[0][0],y=locations[0][1],z=mineral_radius,radius =mineral_radius,color = (1,1,0) )
        #self.silver_mineral_1 = visual.sphere(x=locations[1][0],y=locations[1][1],z=mineral_radius,radius =mineral_radius,color = self.silver)
        #self.silver_mineral_2 = visual.sphere(x=locations[2][0],y=locations[2][1],z=mineral_radius,radius =mineral_radius,color = self.silver)

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
                
        self.init_position()
        
        if self.random_lighting:
            self.randomize_lighting()
    
    def init_mineral_list(self):
        #add the silver minerals
        for i in range(self.silver_mineral_num):
            #self.gold_mineral = visual.sphere(x=locations[0][0],y=locations[0][1],z=mineral_radius,radius =mineral_radius,color = (1,1,0) )
            #self.silver_mineral_1 = visual.sphere(x=locations[1][0],y=locations[1][1],z=mineral_radius,radius =mineral_radius,color = self.silver)
            mineral = visual.sphere(x=0,y=0,z=self.mineral_radius,radius = self.mineral_radius,color= self.silver)
            mineral.collided = False
            mineral.type = Mineral.SILVER
            self.minerals.append(mineral)
        #add a single gold mineral
        mineral = visual.sphere(x=0,y=0,z=self.mineral_radius,radius = self.mineral_radius,color= (1,1,0))
        mineral.collided = False
        mineral.type = Mineral.GOLD
        self.minerals.append(mineral)

        
    
    def reset_minerals(self,exclude_zone = 4):
        num_minerals = len(self.minerals)
        zone_list = list(range(9))
        if exclude_zone != -1:
            zone_list.remove(exclude_zone)
        random_sample = random.sample(zone_list,num_minerals)
        location_list = self.zones[random_sample]
        for i,mineral in enumerate(self.minerals):
            mineral.x = location_list[i][0]
            mineral.y = location_list[i][1]
            mineral.zone = random_sample[i]
            mineral.collided = False
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
            
    def calculate_collision_box(self):
        point_distance = self.point_distance
        shift_distance = point_distance / np.sqrt(2)
        pos,focal = mlab.move()
        v = focal - pos
        v = np.asarray((v[0],v[1]))
        v /= np.linalg.norm(v)
        w = np.asarray((-v[1],v[0])) # used for sideshift
        pos = np.asarray((pos[0],pos[1])) + -v * shift_distance
        x = pos[0]
        y = pos[1]
        angle = mlab.view()[0] + 180
        box = []
        for i in range(4):
            corner_angle = np.deg2rad(angle + 45 + 90 * i )
            corner_x = point_distance * np.cos(corner_angle) + x
            corner_y = point_distance * np.sin(corner_angle) + y
            box.append([corner_x,corner_y])
        return box
    def init_position(self, view_radius = 5):
        angle = 2 * np.pi * np.random.random()
        x = view_radius * np.cos(angle)
        y = view_radius * np.sin(angle)
        fp = [x,y,self.camera_height]
        mlab.view(focalpoint=fp,distance=view_radius,elevation=-90,azimuth=np.rad2deg(angle))
        #mlab.show()
    
        
    def move_position(self,left = 0,right = 0,forwards = 0,backwards = 0, pos_angle = 0, neg_angle = 0, set_value = True):
        mlab.move(forwards - backwards, right - left,0)
        mlab.yaw(pos_angle - neg_angle)
        
    def action_space(self):
        return len(self.actions)
    
    def check_collision(self, mineral, scale = 1):
        box = self.calculate_collision_box()
        radius = self.mineral_radius * scale
        x = mineral.x
        y = mineral.y
        points = []
        for i in range(6):
            angle = i * np.pi / 3
            point = (radius * np.cos(angle) + x, radius * np.sin(angle) + y)
            points.append(point)
        polygon_mineral = Polygon(points)
        polygon_robot = Polygon([tuple(box[0]),tuple(box[1]),tuple(box[2]),tuple(box[3])])
        return polygon_robot.intersects(polygon_mineral)
    def check_collisions(self, scale = 1):
        for mineral in self.minerals:
            if(self.check_collision(mineral,scale)):
                if mineral.type == Mineral.GOLD:
                    return State.GOLD_COLLISION, mineral.zone
                else:
                    return State.SILVER_COLLISION, mineral.zone
        #no collisions found
        return State.STANDARD, -1
    
    def in_stationary_zone(self):
        if self.check_collisions(scale = self.normal_scale)[0] == State.STANDARD and self.check_collisions(scale = self.stationary_scale)[0] == State.GOLD_COLLISION:
            return True, self.check_collisions(scale = self.stationary_scale)[1]
        else:
            return False, -1
    def check_illegal_move(self, x = None, y = None):
        pos = mlab.move()[0]
        if x is None:
            x = pos[0]
        if y is None:
            y = pos[1]
        apothem = self.square_width * 3 / 2
        if x > apothem - 1.5 or x < -apothem + 1.5 or y > apothem - 1.5 or y < -apothem + 1.5:
            return State.ILLEGAL
        else:
            return State.STANDARD
    def check_wall_collision(self):
        box = self.calculate_collision_box()
        apothem = self.square_width * 3 / 2
        for point in box:
            x = point[0]
            y = point[1]
            if x > apothem or x < -apothem or y > apothem or y < -apothem:
                return State.WALL_COLLISION
        return State.STANDARD
    
    def legal_actions(self):
        actions = [0] * self.action_space()
        #abbreviate actions_index_dict to d
        d = self.actions_index_dict
        
        #Set turning to be legal if they are availabe actions, and staying
        if d[Action.CW] != None:
            actions[d[Action.CW]] = 1
        if d[Action.CCW] != None:
            actions[d[Action.CCW]] = 1
        if d[Action.STAY] != None:
            actions[d[Action.STAY]] = 1
            
        pos,focal = mlab.move()
        v = focal - pos
        v = np.asarray((v[0],v[1]))
        v /= np.linalg.norm(v)
        w = np.asarray((-v[1],v[0]))
        pos = np.asarray((pos[0],pos[1]))
        
        #left
        if self.check_illegal_move(*(self.move_distance * w + pos)) != State.ILLEGAL and d[Action.LEFT] != None:
            actions[d[Action.LEFT]] = 1
        #right
        if self.check_illegal_move(*(self.move_distance * -w + pos)) != State.ILLEGAL and d[Action.RIGHT] != None:
            actions[d[Action.RIGHT]] = 1
        #forwards
        if self.check_illegal_move(*(self.move_distance * v + pos)) != State.ILLEGAL and d[Action.FORWARDS] != None:
            actions[d[Action.FORWARDS]] = 1
        #backwards
        if self.check_illegal_move(*(self.move_distance * -v + pos)) != State.ILLEGAL and d[Action.BACKWARDS] != None:
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
        elif action_name != Action.STAY:
            moves[action_name.value - 1] = self.move_distance
        #store previous position
        previous_pos = mlab.move()[0]
        #transition to new state
        self.move_position(*moves)
        new_pos = mlab.move()[0]
        
        #get the reward
        mineral_state, zone = self.check_collisions(self.normal_scale)
        wall_state = self.check_wall_collision()
        game_state = State.STANDARD
        stationary,stationary_zone  = self.in_stationary_zone()
        if stationary:
            self.stationary_count += 1
        else: 
            self.stationary_count = 0
            
        assert (self.check_illegal_move() != State.ILLEGAL), "transitioned to an illegal state with action {} and distance".format(action,self.move_distance)
        
        reward = 0
        
        if wall_state == State.WALL_COLLISION:
            reward += self.loss_reward / 2
            
        if mineral_state == State.GOLD_COLLISION or mineral_state == State.SILVER_COLLISION:
            assert(zone != -1), "invalid zone assignment"
            self.exclude_zone = zone
            reward += self.loss_reward
            game_state = State.LOSS
            done = True
        else:
            done = False
            #calculate previous and current distances
        
            
            if self.reward == Reward.RELATIVE_PROPORTIONAL:
                def distance(x1,y1,x2,y2):
                    return np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                gold_mineral = None
                
                for mineral in self.minerals:
                    if mineral.type == Mineral.GOLD:
                        gold_mineral = mineral
                        
                previous_distance = distance(previous_pos[0],previous_pos[1],gold_mineral.x,gold_mineral.y)
                current_distance = distance(new_pos[0],new_pos[1],gold_mineral.x,gold_mineral.y)
                
                reward += previous_distance - current_distance
            else:
                reward += self.move_reward
                
            if stationary:
                reward += self.win_reward
            
            if self.stationary_count == self.stationary_win_count:
               assert(stationary_zone != -1), "invalid stationary_zone assignment"
               self.exclude_zone = stationary_zone
               game_state = State.WIN
               done = True
               
            #also end the game if there are no more legal actions in the new state 
            if max(self.legal_actions()) == 0:
               self.exclude_zone = 4
               done = True
            
        next_state = self.screenshot()
        
        return next_state, reward, done, game_state
    
    #used for debugging collission, draws spheres representing the collision box
    def display(self):
        pos = mlab.move()[0]
        for i,point in enumerate(self.calculate_collision_box()):
            visual.sphere(x=point[0],y=point[1],z=2.75/2,color=((i*80)/256,1,0),radius=2.75/2)
        visual.sphere(x=pos[0],y=pos[1],z=2.75/2,color=(0,0,1),radius=2.75/2)
        
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
        #modify this line, needs to ignore the collided zone
        self.stationary_count = 0
        
        self.reset_minerals(exclude_zone = self.exclude_zone)        
        self.exclude_zone = -1
        
        self.randomize_colors()
        
        if self.random_lighting:
            self.randomize_lighting()
        
        #self.init_position()

        return self.screenshot() 
    
    def full_reset(self):
        self.exclude_zone = 4
        ss = self.reset()
        self.init_position()
        return ss
