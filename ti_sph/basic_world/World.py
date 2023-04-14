import taichi as ti
from ..basic_op.type import *

@ti.data_oriented
class World:
    def __init__(self):
        self.dim = val_i(3)
        self.space_lb = vec3_f([-8,-8,-8])
        self.space_rt = vec3_f([8,8,8])

        self.gravity = vec3_f([0,-9.8,0])
        self.dt = val_f(0.001)
        self.part_size = val_f(0.1)
        self.avg_neighb_num = val_i(32)

        self.space_size = self.space_rt - self.space_lb
        self.space_center = (self.space_rt + self.space_lb) / 2

        self.part_volume = self.part_size ** self.dim
        self.support_radius = self.part_size * 2
        
