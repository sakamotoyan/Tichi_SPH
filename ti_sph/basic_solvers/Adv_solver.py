import taichi as ti
from.sph_funcs import *
from ..basic_op.type import *
from ..basic_obj.Obj_Particle import Particle

@ti.data_oriented
class Adv_slover:
    def __init__(self, obj: Particle):
        self.obj = obj
        self.dt = obj.m_world.g_dt
        self.gravity = obj.m_world.g_gravity
        self.dim = obj.m_world.g_dim

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
    
    # @ti.kernel
    # def acc2vel_adv(self, out_vel_adv: ti.template()):
    #     for i in range(self.obj.ti_get_stack_top()[None]):
    #         out_vel_adv[i] += self.obj.acc[i] * self.dt[None]
    
    @ti.kernel
    def add_acc_gravity(self):
        for i in range(self.obj.ti_get_stack_top()[None]):
            self.obj.acc[i] += self.gravity[None]

    @ti.kernel
    def acc2vel_adv(self):
        for i in range(self.obj.ti_get_stack_top()[None]):
            self.obj.vel_adv[i] = self.obj.acc[i] * self.dt[None] + self.obj.vel[i]

    @ti.kernel
    def vel_adv2vel(self):
        for i in range(self.obj.ti_get_stack_top()[None]):
            self.obj.vel[i] = self.obj.vel_adv[i]

    @ti.kernel
    def update_pos(self):
        for i in range(self.obj.ti_get_stack_top()[None]):
            self.obj.pos[i] += self.obj.vel[i] * self.dt[None]

    @ti.kernel
    def adv_step(self, in_vel: ti.template(), out_vel_adv: ti.template()):
        for i in range(self.obj.ti_get_stack_top()[None]):
            out_vel_adv[i] = in_vel[i]
            self.obj.acc[i] = self.clean_acc
            self.obj.acc[i] += self.gravity[None]
            out_vel_adv[i] += self.obj.acc[i] * self.dt[None]
