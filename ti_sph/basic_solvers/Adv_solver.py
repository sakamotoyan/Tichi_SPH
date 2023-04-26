import taichi as ti
from.sph_funcs import *
from ..basic_op.type import *
from ..basic_obj.Particle import Particle

@ti.data_oriented
class Adv_slover:
    def __init__(self, obj: Particle):
        self.obj = obj
        self.dt = obj.world.dt
        self.gravity = obj.world.gravity
        self.dim = obj.world.dim

        self.clean_acc = vecxf(self.dim[None])(0)
    
    @ti.kernel
    def clear_acc(self):
        for i in range(self.obj.ti_get_stack_top()[None]):
            self.obj.acc[i] = self.clean_acc

    @ti.kernel
    def add_gravity_acc(self):
        for i in range(self.obj.ti_get_stack_top()[None]):
            self.obj.acc[i] += self.gravity[None]
    
    @ti.kernel
    def add_vis_acc(self, vis_coeff: ti.template()):
        pass
    
    @ti.kernel
    def acc2vel_adv(self, out_vel_adv: ti.template()):
        for i in range(self.obj.ti_get_stack_top()[None]):
            out_vel_adv[i] += self.obj.acc[i] * self.dt[None]
    
    @ti.kernel
    def adv_step(self, in_vel: ti.template(), out_vel_adv: ti.template()):
        for i in range(self.obj.ti_get_stack_top()[None]):
            out_vel_adv[i] = in_vel[i]
            self.obj.acc[i] = self.clean_acc
            self.obj.acc[i] += self.gravity[None]
            out_vel_adv[i] += self.obj.acc[i] * self.dt[None]

    @ti.kernel
    def update_pos(self, in_vel: ti.template(), out_pos: ti.template()):
        for i in range(self.obj.ti_get_stack_top()[None]):
            out_pos[i] += in_vel[i] * self.dt[None]