import taichi as ti
from .constructor import *


@ti.data_oriented
class Node:
    def __init__(self, dim, id, part_num, neighb_cell_num, capacity_list):

        self.info = self.node()
        self.info.id[None]=id
        self.info.part_num[None]=part_num
        self.info.stack_top[None]=0
        self.info.neighb_cell_num[None]=neighb_cell_num
        # self.id = 0
        # self.part_num = part_num
        # self.stack_top = 0
        # self.neighb_cell_num = neighb_cell_num
        self.capacity_list = capacity_list
        

        node_construct(self, dim, part_num, capacity_list)
        node_neighb_cell_construct(self, dim, neighb_cell_num, capacity_list)

    def node(self):
        struct_node = ti.types.struct(
            id=ti.i32,
            part_num=ti.i32,
            stack_top=ti.i32,
            neighb_cell_num=ti.i32,
        )
        return struct_node.field(shape=())
