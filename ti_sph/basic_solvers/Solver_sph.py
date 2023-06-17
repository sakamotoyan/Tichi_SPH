import taichi as ti
import math
from ..basic_obj.Obj_Particle import Particle

@ti.data_oriented
class SPH_solver:
    def __init__(self, obj: Particle):
        self.obj = obj

    @ti.kernel
    def loop_neighb(self, neighb_pool:ti.template(), neighb_obj:ti.template(), func:ti.template()):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            neighb_part_num = neighb_pool.neighb_obj_pointer[part_id, neighb_obj.ti_get_id()[None]].size
            neighb_part_shift = neighb_pool.neighb_obj_pointer[part_id, neighb_obj.ti_get_id()[None]].begin
            for neighb_part_iter in range(neighb_part_num):
                neighb_part_id = neighb_pool.neighb_pool_container[neighb_part_shift].neighb_part_id
                ''' Code for Computation'''
                func(part_id, neighb_part_id, neighb_part_shift, neighb_pool, neighb_obj)
                ''' End of Code for Computation'''
                ''' DO NOT FORGET TO COPY/PASE THE FOLLOWING CODE WHEN REUSING THIS FUNCTION '''
                neighb_part_shift = neighb_pool.neighb_pool_container[neighb_part_shift].next
    
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

    @ti.func
    def inloop_accumulate_density(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].W
        self.obj.sph[part_id].density += neighb_obj.mass[neighb_part_id] * cached_W
    
    def sph_compute_density(self, neighb_pool):
        self.obj.clear(self.obj.sph.density)
        for neighb_obj in neighb_pool.neighb_obj_list:
            ''' Compute Density '''
            self.loop_neighb(neighb_pool, neighb_obj, self.inloop_accumulate_density)