import taichi as ti

from .neighb_search import update_pos_in_neighb_search

from ...basic_op.type import *
from ...basic_obj.Obj_Particle import Particle

@ti.data_oriented
class World:
    def __init__(self, dim=3):
        ''' GLOBAL CONFIGURATION '''
        self.g_dim = val_i(dim)
        self.g_space_lb = vecx_f(self.g_dim[None])
        self.g_space_rt = vecx_f(self.g_dim[None])
        self.g_gravity = vecx_f(self.g_dim[None])
        self.g_space_lb.fill(-8)
        self.g_space_rt.fill(8)
        self.g_gravity[None][1] = -9.8
        self.g_dt = val_f(0.001)
        self.g_part_size = val_f(0.1)
        self.g_avg_neighb_part_num = val_i(32)
        self.g_obj_num = val_i(3)
        self.g_sound_speed = val_f(100)

        self.dependent_init()

        self.part_obj_list = []

    def dependent_init(self):
        self.space_size = vecx_f(self.g_dim[None])
        self.space_center = vecx_f(self.g_dim[None])
        self.space_size[None] = self.g_space_rt[None] - self.g_space_lb[None]
        self.space_center[None] = (self.g_space_rt[None] + self.g_space_lb[None]) / 2

        self.part_volume = val_f(self.g_part_size[None] ** self.g_dim[None])
        self.support_radius = val_f(self.g_part_size[None] * 2)

    def refresh(self):
        self.dependent_init()
    
    def set_part_size(self, size):
        self.g_part_size = val_f(size)
        self.refresh()
    
    def set_time_step(self, dt):
        self.g_dt = val_f(dt)

    def add_part_obj(self, part_num, is_dynamic):
        obj = Particle(part_num, self.g_part_size, is_dynamic)
        self.part_obj_list.append(obj)
        obj.set_id(self.part_obj_list.index(obj))
        obj.set_world(self)
        return obj
    
    # Functions: neighbour search
    update_pos_in_neighb_search = update_pos_in_neighb_search
