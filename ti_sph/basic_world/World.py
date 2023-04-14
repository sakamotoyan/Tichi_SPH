import taichi as ti
from ..basic_op.type import *

@ti.data_oriented
class World:
    def __init__(self):
        ''' GLOBAL CONFIGURATION '''
        self.dim = val_i(3)
        self.space_lb = vec3_f([-8,-8,-8])
        self.space_rt = vec3_f([8,8,8])

        self.gravity = vec3_f([0,-9.8,0])
        self.dt = val_f(0.001)
        self.part_size = val_f(0.1)
        self.avg_neighb_part_num = val_i(32)
        self.obj_num = val_i(3)

        self.space_size = vecx_f(self.dim[None])
        self.space_center = vecx_f(self.dim[None])
        self.space_size[None] = self.space_rt[None] - self.space_lb[None]
        self.space_center[None] = (self.space_rt[None] + self.space_lb[None]) / 2

        self.part_volume = val_f(self.part_size[None] ** self.dim[None])
        self.support_radius = val_f(self.part_size[None] * 2)
