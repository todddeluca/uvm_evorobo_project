

import numpy as np


def send_trophy(sim, x, y, z, size, collision_group):
    '''
    z: The vertical position on which to place the statue. The bottom of the statue will be placed at z.
    '''
    # the trophy
    r = 254 / 255
    g = 182 / 255
    b = 37 / 255
    base_id = sim.send_box(x=x, 
                           y=y,
                           z=z + 0.5 * size,
                           length=size * 4, width=size * 4, height=size,
                           r=r, g=g, b=b,
                           collision_group=collision_group)
    stem_id = sim.send_cylinder(x=x, 
                                y=y, 
                                z=z + 1 * size + size * 2,
                                r1=0, r2=0, r3=1,
                                length=size * 4,
                                radius=size,
                                capped=False,
                                r=r, g=g, b=b,
                                collision_group=collision_group)
    bowl_id = sim.send_sphere(x=x,
                              y=y, 
                              z=z + 1 * size + size * 4 + size * 2,
                              radius=size * 2,
                              r=r, g=g, b=b,
                              collision_group=collision_group)
    sim.send_light_source(body_id=stem_id)
    for id in (base_id, stem_id, bowl_id):
        sim.send_fixed_joint(id, -1)


class SpatialScaffoldingStairsEnv:
    '''
    Floating Stairs with a light source trophy on top.
    
    Each stair is placed vertically relative to the previous stair. This is the
    rise of the stair. The rise of each stair increases from the first stair to
    the last stair according to the scaffolding schedule. This spatial change in rise
    is where the term "Spatial Scaffolding" comes from.
    
    The rise of the final stair is always equal to the max_rise.
    '''
    def __init__(self, num_stairs, depth, width, thickness, y_offset, 
                 max_rise, temp=0, min_temp=-10, temp_scale=4):
        '''
        y_offset: y-axis position of the front of the initial stair.
        max_rise: the maximum rise from one stair to the next.
        rise_rate: the scheduling parameter from the genome which determines the 
          spatial scaffolding rate.
        temp: scaffolding temperature. Expected to be roughly in the range -1 to 1, but
          can range from -inf to inf. 0 means medium scaffolding, 
          >= 1 means about max_rise for every stair, and <= -1 means about zero
          rise for every stair except the last.
        min_temp: <= -1 means stairs can rise as little as possible. >= 1 means max_rise
          for every step.
        temp_scale: >= 0. multiplied by temp to generate stairs. larger scale means more extreme
        stairs (either very flat or very steep). 
        '''
        self.num_stairs = num_stairs
        self.depth = depth 
        self.width = width
        self.y_offset = y_offset
        self.thick = thickness # stair thickness
        self.max_rise = max_rise
        self.temp = temp
        self.temp_scale = np.abs(temp_scale)
        self.min_temp = np.clip(min_temp, -10, 10)
        self.group = 'env' # collision group
        
    def send_to(self, sim):
        # normalized stair position (0 to 1)
        positions = np.arange(self.num_stairs) / (self.num_stairs - 1) 
        # the fraction of max_rise for each stair.
        temp = max(self.temp, self.min_temp)
        if temp >= 0:
            fracs = 1 - np.power(1 - positions, np.exp(self.temp_scale * temp))
        else:
            fracs = np.power(positions, np.exp(-self.temp_scale * temp))
        
        rises = fracs * self.max_rise
        
        # x, y, z position of each stair
        stair_coords = [(0, 
                         self.y_offset + 0.5 * self.depth + i * self.depth,
                         0.5 * self.thick + rises[:(i+1)].sum()
                        ) for i in range(self.num_stairs)]
        stair_ids = []
        for x, y, z in stair_coords:
            sid = sim.send_box(x=x, y=y, z=z, 
                               length=self.depth, width=self.width, height=self.thick,
                               r=1, g=1, b=1,
                               collision_group=self.group)
            stair_ids.append(sid)
            
        # fix the stairs in place
        for sid in stair_ids:
            sim.send_fixed_joint(sid, -1)
        
        # the trophy is the light source
        send_trophy(sim, x=0, y=stair_coords[-1][1], z=stair_coords[-1][2] + 0.5 * self.thick,
                    size=self.thick, collision_group=self.group)
        
        self.x_min = -self.width / 2
        self.x_max = self.width / 2 
        
        # dummy touch id for fitness function
        touch_ids = [sim.send_touch_sensor(body_id=stair_ids[-1])]
        return touch_ids

    
class PhototaxisEnv:
    '''
    Environment for testing Phototaxis by placing a light source block on the ground.
    '''
    
    def __init__(self, id_, L):
        self.group = 'env'
        self.id_ = id_
        self.l = L
        self.w = L
        self.h = L
        self.z = L / 2
        
        if id_ == 0: # front
            self.x = 0
            self.y = 30 * L
        elif id_ == 1: # right
            self.x = 30 * L
            self.y = 0
        elif id_ == 2: # back
            self.x = 0
            self.y = -30 * L
        elif id_ == 3: # left
            self.x = -30 * L
            self.y = 0
                    
    def send_to(self, sim):
        light_source = sim.send_box(x=self.x, y=self.y, z=self.z,
                                    length=self.l, width=self.w, height=self.h,
                                    r=0.9, g=0.9, b=0.9,
                                    collision_group=self.group,
                                   )
        sim.send_light_source(body_id=light_source)
        # dummy touch id for fitness function
        touch_ids = [sim.send_touch_sensor(body_id=light_source)]
        return touch_ids
          

class StairsEnv:
    '''Floating stairs, whose top stair is a light source.'''
    def __init__(self, num_stairs, depth, width, thickness, angle, y_offset):
        self.num_stairs = num_stairs
        self.depth = depth 
        self.width = width
        self.y_offset = y_offset
        self.thick = thickness # stair thickness
        self.group = 'env' # collision group
        self.angle = angle # angle of stairs in radians, from 0 to pi/2
        
    def send_to(self, sim):
        rise = np.sin(self.angle) * self.depth
        run = np.cos(self.angle) * self.depth
        
        # x, y, z position of each stair
        stair_coords = [(0, 
                         self.y_offset + 0.5 * self.depth + i * run,
                         i * rise + 0.5 * self.thick
                        ) for i in range(self.num_stairs)]
        stair_ids = []
        for x, y, z in stair_coords:
            sid = sim.send_box(x=x, y=y, z=z, 
                               length=self.depth, width=self.width, height=self.thick,
                               r=1, g=1, b=1,
                               collision_group=self.group)
            stair_ids.append(sid)
            
        # fix the stairs in place
        for sid in stair_ids:
            sim.send_fixed_joint(sid, -1)
        
        # the trophy is the light source
        send_trophy(sim, x=0, y=stair_coords[-1][1], z=stair_coords[-1][2] + 0.5 * self.thick,
                    size=self.thick, collision_group=self.group)
        
        self.x_min = -self.width / 2
        self.x_max = self.width / 2 
        
        # dummy touch id for fitness function
        touch_ids = [sim.send_touch_sensor(body_id=stair_ids[-1])]
        return touch_ids

    
class AngledLatticeEnv:
    '''Constructs a rectangular lattice whose top rung is a light source.'''
    def __init__(self, num_rungs, num_rails, rung_spacing, rail_spacing, thickness, angle, y_offset):
        self.num_rungs = num_rungs
        self.num_rails = num_rails
        self.rung_spacing = rung_spacing
        self.rail_spacing = rail_spacing
        self.thick = thickness # rung thickness/radius
        self.y_offset = y_offset
        self.angle = angle # angle of ladder in radians, from 0 to pi/2
        self.group = 'env' # collision group
        
    def send_to(self, sim):
        # x, y, z coord of center of each rung
        rung_coords = [(0, 
                        (self.y_offset + self.thick + i * np.cos(self.angle) * self.rung_spacing), 
                        (i * np.sin(self.angle) * self.rung_spacing + self.thick)
                       ) for i in range(self.num_rungs)]
        # make rungs
        rung_ids = []
        for x, y, z in rung_coords:
            rung_id = sim.send_cylinder(x=x, y=y, z=z,
                                        r1=1, r2=0, r3=0, 
                                        length=(self.num_rails - 1) * self.rail_spacing,
                                        radius=self.thick,
                                        collision_group=self.group)
            rung_ids.append(rung_id)

        # fix the rungs in place
        for id in rung_ids:
            sim.send_fixed_joint(id, -1)
            
        # x, y, z coord of center of each rail
        rail_coords = [(-1 * (self.num_rails - 1) * self.rail_spacing / 2 + i * self.rail_spacing,
                        (rung_coords[0][1] + rung_coords[-1][1]) / 2, 
                        (rung_coords[0][2] + rung_coords[-1][2]) / 2
                       ) for i in range(self.num_rails)]
        
        # make rails
        rail_ids = []
        for x, y, z in rail_coords:
            rail_id = sim.send_cylinder(x=x, y=y, z=z,
                                        r1=0, r2=np.cos(self.angle), r3=np.sin(self.angle), 
                                        length=(self.num_rungs - 1) * self.rung_spacing,
                                        radius=self.thick,
                                        collision_group=self.group)
            rail_ids.append(rail_id)

        # fix the rails in place
        for id in rail_ids:
            sim.send_fixed_joint(id, -1)
        
        # the trophy
        send_trophy(sim, x=0, y=rung_coords[-1][1], z=rung_coords[-1][2] + self.thick,
                    size=self.thick, collision_group=self.group)
        
        self.x_min = rail_coords[0][0]
        self.x_max = rail_coords[-1][0]
        
        # dummy touch id for fitness function
        touch_ids = [sim.send_touch_sensor(body_id=rung_ids[-1])]
        return touch_ids


class AngledLadderEnv:
    '''Constructs a ladder whose top rung is a light source.'''
    def __init__(self, num_rungs, spacing, width, thickness, angle, y_offset):
        self.num_rungs = num_rungs
        self.spacing = spacing # rung spacing
        self.width = width
        self.thick = thickness # rung thickness/radius
        self.y_offset = y_offset
        self.angle = angle # angle of ladder in radians, from 0 to pi/2
        self.rise = np.sin(self.angle) * self.spacing
        self.run = np.cos(self.angle) * self.spacing
        self.group = 'env' # collision group
        
    def send_to(self, sim):
        rung_ids = []
        for i in range(self.num_rungs + 1):
            # a small, golden top rung
            length = self.width if i < self.num_rungs else self.thick
            r = 1 if i < self.num_rungs else 254 / 255
            g = 1 if i < self.num_rungs else 182 / 255
            b = 1 if i < self.num_rungs else 37 / 255
            r_id = sim.send_cylinder(x=0, 
                                     y=(self.y_offset + self.thick + i * self.run), 
                                     z=(i * self.rise + self.thick),
                                     r1=1, r2=0, r3=0, 
                                     length=length, radius=self.thick,
                                     r=r, g=g, b=b,
                                     collision_group=self.group)
            rung_ids.append(r_id)

        # fix the rungs in place
        for r_id in rung_ids:
            sim.send_fixed_joint(r_id, -1)
        
        # last "rung" is the light source
        sim.send_light_source(body_id=rung_ids[-1])
        
        self.x_min = -self.width / 2
        self.x_max = self.width / 2 
        
        # dummy touch id for fitness function
        touch_ids = [sim.send_touch_sensor(body_id=rung_ids[-1])]
        return touch_ids

    
class LadderEnv:
    '''Constructs a ladder whose top rung is a light source.'''
    def __init__(self, length, width, thickness, spacing, y_offset):
        self.length = length 
        self.width = width
        self.y_offset = y_offset
        self.thick = thickness # rung thickness
        self.spacing = spacing # rung spacing
        self.group = 'env' # collision group
        
    def send_to(self, sim):

        # Rails of the ladder
        # x, y, z, r1, r2, r3, l, r
        left_rail = (-self.width / 2, self.y_offset, self.length / 2 + self.thick, # account for cylinder end cap
                     0, 0, 1,
                     self.length, self.thick)
        right_rail = (self.width / 2, self.y_offset, self.length / 2 + self.thick,
                     0, 0, 1,
                      self.length, self.thick)
        
        rails = [left_rail, right_rail]
        rail_ids = []
        for x, y, z, r1, r2, r3, l, r in rails:
            id_ = sim.send_cylinder(x=x, y=y, z=z, 
                                  r1=r1, r2=r2, r3=r3, 
                                  length=l, radius=r,
                                  r=0.9, g=0.9, b=0.9,
                                  collision_group=self.group)
            rail_ids.append(id_)
                        
        # rungs: x, y, z, r1, r2, r3, l, w, h
        # make n rungs along the lengthe of the rail, separated by spacing
        rungs = []
        rung_ids = []
        touch_ids = [] # touch sensor ids
        pos = 0 + self.thick + self.thick / 2 # including cylinder cap of rail
        top = self.length
        while pos < top:
            rungs.append((0, self.y_offset, pos,
                          1, 0, 0, # ladder is oriented along x-axis
                          self.width, self.thick,
                         ))
            pos += self.spacing
        for x, y, z, r1, r2, r3, l, r in rungs:
            id_ = sim.send_cylinder(x=x, y=y, z=z, 
                                  r1=r1, r2=r2, r3=r3, 
                                  length=l, radius=r,
                                  r=0.9, g=0.9, b=0.9,
                                  collision_group=self.group)
            rung_ids.append(id_)
            # rung touch sensor
            tid = sim.send_touch_sensor(body_id=id_)
            touch_ids.append(tid)

        # fix the rungs to the rails
        for rid in rung_ids:
            sim.send_fixed_joint(rid, rail_ids[0])
            sim.send_fixed_joint(rid, rail_ids[1])
        
        # fix the rails to the world
        sim.send_fixed_joint(rail_ids[0], -1)
        
        # top rung is the goal / light source
        sim.send_light_source(body_id=rung_ids[-1])

        return touch_ids


