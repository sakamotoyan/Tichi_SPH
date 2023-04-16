import taichi as ti
import numpy as np

from .generator import *
from ..basic_obj.Particle import Particle

@ti.data_oriented
class Box_generator(Data_generator):
    def __init__(self, obj: Particle, lb: ti.Vector, rt: ti.Vector, layers: ti.i32):
        self.obj = obj
        self.lb = lb
        self.rt = rt
        self.layers = layers
        self.dim = ti.static(lb.n)
    
    def push_pos(self, factor: float = 1.0) -> ti.i32:
        span = factor * self.obj.part_size[None]
        pushed_part_num = self.ker_push_box_based_on_span(self.obj.pos, self.obj.get_stack_top(), span)
        if pushed_part_num+self.obj.get_stack_top()[None] > self.obj.pos.shape[0]:
            raise Exception("ERROR from push_box(): overflow")
        return pushed_part_num

    ''' This function is basically a part of ker_push_box_based_on_span, but it is separated for debugging purpose. '''
    ''' I did this because I am lazy to write a new kernel for debugging.'''
    @ti.kernel
    def pushed_num_preview(self, factor: ti.f32) -> ti.i32:
        span = factor * self.obj.part_size[None]
        current_node_num = 0
        pushed_node_seq_coder = ti.Vector([0, 0, 0])

        pushed_node_seq = int(ti.ceil((self.rt - self.lb) / span))
        pushed_node_seq_offset = int(ti.ceil((self.rt - self.lb) / span)) + (self.layers * 2)

        for i in ti.static(range(self.dim)):
            if pushed_node_seq[i] == 0:
                pushed_node_seq[i] = 1  # at least push one

        tmp = 1
        for i in ti.static(range(self.dim)):
            pushed_node_seq_coder[i] = tmp
            tmp *= pushed_node_seq_offset[i]

        pushed_node_num = 1
        pushed_node_num_solid = 1
        for i in ti.static(range(self.dim)):
            pushed_node_num *= pushed_node_seq[i]
            pushed_node_num_solid *= pushed_node_seq_offset[i]
        pushed_node_num = pushed_node_num_solid - pushed_node_num
        new_node_num = current_node_num + pushed_node_num

        inc = ti.Vector([current_node_num])
        for i in range(pushed_node_num_solid):
            tmp = i
            a = self.rt - self.lb
            flag = False
            dim_check = 0
            for j in ti.static(range(self.dim - 1, -1, -1)):
                a[j] = tmp // pushed_node_seq_coder[j]
                tmp = tmp % pushed_node_seq_coder[j]
            t1 = (a - 1) - pushed_node_seq
            t2 = a - self.layers
            
            for i in ti.static(range(t1.n)):
                if t1[i] > 0:
                    flag = True
            for i in ti.static(range(t2.n)):
                if t2[i] < 0:
                    flag = True
            if flag:
                index = ti.atomic_add(inc[0], 1)
        return inc[0]

    @ti.kernel
    def ker_push_box_based_on_span(self, obj_pos:ti.template(), obj_stack_top: ti.template(), span: ti.f32) -> ti.i32:
        current_node_num = obj_stack_top[None]
        pushed_node_seq_coder = ti.Vector([0, 0, 0])

        pushed_node_seq = int(ti.ceil((self.rt - self.lb) / span))
        pushed_node_seq_offset = int(ti.ceil((self.rt - self.lb) / span)) + (self.layers * 2)

        for i in ti.static(range(self.dim)):
            if pushed_node_seq[i] == 0:
                pushed_node_seq[i] = 1  # at least push one

        tmp = 1
        for i in ti.static(range(self.dim)):
            pushed_node_seq_coder[i] = tmp
            tmp *= pushed_node_seq_offset[i]

        pushed_node_num = 1
        pushed_node_num_solid = 1
        for i in ti.static(range(self.dim)):
            pushed_node_num *= pushed_node_seq[i]
            pushed_node_num_solid *= pushed_node_seq_offset[i]
        pushed_node_num = pushed_node_num_solid - pushed_node_num
        new_node_num = current_node_num + pushed_node_num

        inc = ti.Vector([current_node_num])
        for i in range(pushed_node_num_solid):
            tmp = i
            a = self.rt - self.lb
            flag = False
            dim_check = 0
            for j in ti.static(range(self.dim - 1, -1, -1)):
                a[j] = tmp // pushed_node_seq_coder[j]
                tmp = tmp % pushed_node_seq_coder[j]
            t1 = (a - 1) - pushed_node_seq
            t2 = a - self.layers
            
            for i in ti.static(range(t1.n)):
                if t1[i] > 0:
                    flag = True
            for i in ti.static(range(t2.n)):
                if t2[i] < 0:
                    flag = True
            if flag:
                index = ti.atomic_add(inc[0], 1)
                for j in ti.static(range(self.dim - 1, -1, -1)):
                    obj_pos[index][j] = (a[j] - self.layers) * span + self.lb[j]
        return inc[0]