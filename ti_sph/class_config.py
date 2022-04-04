import taichi as ti
from .constructor import *

@ti.data_oriented
class Config:
    def __init__(self, dim, capacity_list):
        info_construct(self, dim, capacity_list)
        
    @ti.kernel
    def calculate_neighb_cell_param(self):
        struct_space = ti.static(self.space)
        struct_neighb = ti.static(self.neighb)
        
        struct_neighb.cell_num_vec[None] = ti.ceil(
            (struct_space.rt[None] - struct_space.lb[None]) / struct_neighb.cell_size[None])
        
        for i in ti.static(range(struct_neighb.cell_coder[None].n)):
            struct_neighb.cell_coder[None][i] = 1
        struct_neighb.cell_num[None] = 1
        
        for i in ti.static(range(struct_neighb.cell_coder[None].n)):
            struct_neighb.cell_coder[None][i] = struct_neighb.cell_num_vec[None][i]
            struct_neighb.cell_num[None] *= int(
                struct_neighb.cell_num_vec[None][i])