import taichi as ti
from.SPH_funcs import *
from ..basic_op.type import *

@ti.data_oriented
class Adv_funcs:
    def __init__(self, obj__: ti.template(), dt_: ti.template(), g_: ti.template(), vis_coeff_k_: ti.template()):
        self.obj__ = obj__
        self.dt_ = dt_
        self.g_ = g_
        self.vis_coeff_k_ = vis_coeff_k_
        self.dim = ti.static(obj__.pos_.n)

        self.clean_acc = vecxf(self.dim)(0)
    
    @ti.kernel
    def clear_acc(self):
        for i in range(self.obj__.stack_top_[None]):
            self.obj__.acc_[i] = self.clean_acc

    @ti.kernel
    def add_gravity_acc(self):
        for i in range(self.obj__.stack_top_[None]):
            self.obj__.acc_[i] += self.g_
    
    @ti.kernel
    def add_vis_acc(self, vis_coeff: ti.template()):
        pass
    
    @ti.kernel
    def acc2vel_adv(self, out_vel_adv_: ti.template()):
        for i in range(self.obj__.stack_top_[None]):
            out_vel_adv_[i] += self.obj__.acc_[i] * self.dt_[None]
    
    @ti.kernel
    def adv_step(self, in_vel_: ti.template(), out_vel_adv_: ti.template()):
        for i in range(self.obj__.stack_top_[None]):
            out_vel_adv_[i] = in_vel_[i]
            self.obj__.acc_[i] = self.clean_acc
            self.obj__.acc_[i] += self.g_[None]
            out_vel_adv_[i] += self.obj__.acc_[i] * self.dt_[None]

    @ti.kernel
    def update_pos(self, in_vel_: ti.template(), out_pos_: ti.template()):
        for i in range(self.obj__.stack_top_[None]):
            out_pos_[i] += in_vel_[i] * self.dt_[None]