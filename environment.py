
import constants as c


class Environment:
    
    def __init__(self, id_):
        self.id_ = id_
        self.l = c.L
        self.w = c.L
        self.h = c.L
        self.z = c.L / 2
        
        if id_ == 0: # front
            self.x = 0
            self.y = 30 * c.L
        elif id_ == 1: # right
            self.x = 30 * c.L
            self.y = 0
        elif id_ == 2: # back
            self.x = 0
            self.y = -30 * c.L
        elif id_ == 3: # left
            self.x = -30 * c.L
            self.y = 0
                    
    def send_to(self, sim):
        light_source = sim.send_box(x=self.x, y=self.y, z=self.z,
                                    length=self.l, width=self.w, height=self.h,
                                    r=0.9, g=0.9, b=0.9)
        sim.send_light_source(body_id=light_source)
