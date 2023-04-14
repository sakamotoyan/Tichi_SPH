import taichi as ti
import math
from.SPH_funcs import *
from ..basic_op.type import *
from ..basic_obj.Particle import Particle

@ti.data_oriented
class DF_solver:
    def __init__(self, obj: Particle, incompressible_threshold: ti.f32 = 1e-4, div_free_threshold: ti.f32 = 1e-3, incompressible_iter_max: ti.i32 = 50, div_free_iter_max: ti.i32 = 50):
        self.obj=obj
        self.dt=obj.world.dt
        self.incompressible_threshold_ = val_f(incompressible_threshold)
        self.div_free_threshold_ = val_f(div_free_threshold)
        self.incompressible_iter_max_ = val_i(incompressible_iter_max)
        self.div_free_iter_max_ = val_i(div_free_iter_max)

        self.compressible_ratio_ = val_f(1)
        self.div_free_ratio_ = val_f(1)
        self.incompressible_iter_ = val_i(0)
        self.div_free_iter_ = val_i(0)
        self.inv_dt_ = val_f(1 / self.dt[None])
        self.neg_inv_dt_ = val_f(-1 / self.dt[None])
        self.dim = obj.world.dim
        sig_dim = self.sig_dim(self.dim[None])
        self.compute_sig(sig_dim)
        

    ''' [NOTICE] If sig_dim is decorated with @ti.func, and called in a kernel, 
    it will cause a computation error due to the use of math.pi. This bug is tested. '''
    def sig_dim(self, dim):
        sig = 0
        if dim == 3:
            sig = 8 / math.pi 
        elif dim == 2:
            sig = 40 / 7 / math.pi
        elif dim == 1:
            sig = 4 / 3
        return sig
    
    @ti.kernel
    def compute_sig(self, sig_dim: ti.f32):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            self.obj.sph[part_id].h = self.obj.size[part_id] * 2
            self.obj.sph[part_id].sig = sig_dim / ti.pow(self.obj.sph[part_id].h, self.dim[None])
            self.obj.sph[part_id].sig_inv_h = self.obj.sph[part_id].sig / self.obj.sph[part_id].h

    @ti.kernel
    def loop_neighb(self, neighb_list__:ti.template(), neighb_obj__:ti.template(), func:ti.template()):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            neighb_part_num = neighb_list__.neighb_obj_pointer[part_id, neighb_obj__.ti_get_id()[None]].size
            neighb_part_shift = neighb_list__.neighb_obj_pointer[part_id, neighb_obj__.ti_get_id()[None]].begin
            for neighb_part_iter in range(neighb_part_num):
                neighb_part_id = neighb_list__.neighb_pool_container[neighb_part_shift].neighb_part_id
                ''' Code for Computation'''
                func(part_id, neighb_part_id, neighb_part_shift, neighb_list__, neighb_obj__)
                ''' End of Code for Computation'''
                ''' DO NOT FORGET TO COPY/PASE THE FOLLOWING CODE WHEN REUSING THIS FUNCTION '''
                neighb_part_shift = neighb_list__.neighb_pool_container[neighb_part_shift].next

    @ti.func
    def inloop_compute_u_density(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_list__:ti.template(), neighb_obj__:ti.template()):
        cached_W = neighb_list__.cached_neighb_attributes[neighb_part_shift].W
        self.obj.sph[part_id].density += neighb_obj__.mass[neighb_part_id] * cached_W
    
    @ti.func
    def inloop_compute_u_alpha_1_2(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_list__:ti.template(), neighb_obj__:ti.template()):
        cached_dist = neighb_list__.cached_neighb_attributes[neighb_part_shift].dist
        cached_grad_W = neighb_list__.cached_neighb_attributes[neighb_part_shift].grad_W
        if bigger_than_zero(cached_dist):
            self.obj.sph_df[part_id].alpha_1 += neighb_obj__.mass[neighb_part_id] * cached_grad_W
            self.obj.sph_df[part_id].alpha_2 += cached_grad_W.dot(cached_grad_W)

    @ti.func
    def inloop_compute_u_alpha_1(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_list__:ti.template(), neighb_obj__:ti.template()):
        cached_dist = neighb_list__.cached_neighb_attributes[neighb_part_shift].dist
        cached_grad_W = neighb_list__.cached_neighb_attributes[neighb_part_shift].grad_W
        if bigger_than_zero(cached_dist):
            self.obj.sph_df[part_id].alpha_1 += neighb_obj__.mass[neighb_part_id] * cached_grad_W

    @ti.func
    def inloop_compute_u_alpha_2(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_list__:ti.template(), neighb_obj__:ti.template()):
        cached_dist = neighb_list__.cached_neighb_attributes[neighb_part_shift].dist
        cached_grad_W = neighb_list__.cached_neighb_attributes[neighb_part_shift].grad_W
        if bigger_than_zero(cached_dist):
            self.obj.sph_df[part_id].alpha_2 += cached_grad_W.dot(cached_grad_W) * neighb_obj__.mass[neighb_part_id]

    @ti.kernel
    def compute_a_alpha(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            self.obj.sph_df[part_id].alpha = self.obj.sph_df[part_id].alpha_1.dot(self.obj.sph_df[part_id].alpha_1) / self.obj.mass[part_id] + self.obj.sph_df[part_id].alpha_2
            if not bigger_than_zero(self.obj.sph_df[part_id].alpha):
                self.obj.sph_df[part_id].alpha = make_bigger_than_zero()

    @ti.kernel
    def compute_a_delta_density(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            self.obj.sph_df[part_id].delta_density = self.obj.sph[part_id].density - self.obj.rest_density[part_id]
    
    @ti.kernel
    def ReLU_a_delta_density(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            if self.obj.sph_df[part_id].delta_density < 0:
                self.obj.sph_df[part_id].delta_density = 0

    @ti.func
    def inloop_compute_u_delta_density_from_vel_adv(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_list__:ti.template(), neighb_obj__:ti.template()):
        cached_dist = neighb_list__.cached_neighb_attributes[neighb_part_shift].dist
        cached_grad_W = neighb_list__.cached_neighb_attributes[neighb_part_shift].grad_W
        if bigger_than_zero(cached_dist):
            self.obj.sph_df[part_id].delta_density += cached_grad_W.dot(self.obj.sph_df[part_id].vel_adv-neighb_obj__.sph_df[neighb_part_id].vel_adv) * neighb_obj__.mass[neighb_part_id] * self.dt[None]

    @ti.func
    def inloop_compute_u_vel_adv_from_alpha(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_list__:ti.template(), neighb_obj__:ti.template()):
        cached_dist = neighb_list__.cached_neighb_attributes[neighb_part_shift].dist
        cached_grad_W = neighb_list__.cached_neighb_attributes[neighb_part_shift].grad_W
        if bigger_than_zero(cached_dist):
            self.obj.sph_df[part_id].vel_adv += self.neg_inv_dt_[None] * cached_grad_W / self.obj.mass[part_id] \
                * ((self.obj.sph_df[part_id].delta_density * neighb_obj__.mass[neighb_part_id] / self.obj.sph_df[part_id].alpha) \
                   + (neighb_obj__.sph_df[neighb_part_id].delta_density * self.obj.mass[part_id] / neighb_obj__.sph_df[neighb_part_id].alpha))

    @ti.kernel 
    def update_compressible_ratio(self):
        self.compressible_ratio_[None] = 0
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            self.compressible_ratio_[None] += self.obj.sph_df[part_id].delta_density / self.obj.rest_density[part_id]
        self.compressible_ratio_[None] /= self.obj.ti_get_stack_top()[None]

    @ti.kernel
    def update_vel(self, out_vel_: ti.template()):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            out_vel_[part_id] = self.obj.sph_df[part_id].vel_adv

    @ti.kernel
    def get_vel_adv(self, in_vel_adv: ti.template()):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            self.obj.sph_df[part_id].vel_adv = in_vel_adv[part_id]

    def df_step_static_phase(self, neighb_list:ti.template()):
        self.obj.clear(self.obj.sph.density)
        self.obj.clear(self.obj.sph_df.alpha_1)
        self.obj.clear(self.obj.sph_df.alpha_2)

        self.div_free_iter_[None] = 0
        self.incompressible_iter_[None] = 0

        for neighb_obj__ in neighb_list.neighb_obj_list:
            ''' Compute Density '''
            self.loop_neighb(neighb_list, neighb_obj__, self.inloop_compute_u_density)
            ''' Compute Alpha_1, Alpha_2 ''' 
            if self.obj.is_dynamic:
                self.loop_neighb(neighb_list, neighb_obj__, self.inloop_compute_u_alpha_1)
                if neighb_obj__.is_dynamic:
                    self.loop_neighb(neighb_list, neighb_obj__, self.inloop_compute_u_alpha_2)
            else: 
                if neighb_obj__.is_dynamic:
                    self.loop_neighb(neighb_list, neighb_obj__, self.inloop_compute_u_alpha_2)

        ''' Compute Alpha '''
        self.compute_a_alpha()

    def df_step_dynamic_phase(self, in_vel_adv_: ti.template(), out_vel_: ti.template(), neighb_list__:ti.template()):
        ''' Compute Delta Density '''
        self.compute_a_delta_density()    

    def df_step_dynamic_phase(self, in_vel_adv: ti.template(), out_vel_: ti.template(), neighb_list__:ti.template()):
        self.get_vel_adv(in_vel_adv)
        while True:
            self.incompressible_iter_[None] += 1

            ''' Compute Delta Density '''
            self.compute_a_delta_density()
            for neighb_obj__ in neighb_list__.neighb_obj_list:
                ''' Further Update Delta Density '''
                self.loop_neighb(neighb_list__, neighb_obj__, self.inloop_compute_u_delta_density_from_vel_adv)
            
            ''' Update Density Ratio from Delta Density '''
            self.update_compressible_ratio()
            ''' Incompressible Condition '''
            if not (self.compressible_ratio_[None] > self.incompressible_threshold_[None] \
                    and self.incompressible_iter_[None] < self.incompressible_iter_max_[None]):
                break
            
            for neighb_obj__ in neighb_list__.neighb_obj_list:
                ''' Further Update Delta Density '''
                self.loop_neighb(neighb_list__, neighb_obj__, self.inloop_compute_u_vel_adv_from_alpha)

        self.update_vel(out_vel_)
        print(self.incompressible_iter_[None])

def co_df(df_solvers):
    pass

            
@ti.data_oriented
class DF_wrap:
    def __init__(self, DF_solvers):
        self.solver_list = []
        for solver in DF_solvers:
            '''check if solver is DF_solver'''
            if not isinstance(solver, DF_solver):
                raise TypeError('DF_wrap only accepts DF_solver')
            self.solver_list.append(solver)