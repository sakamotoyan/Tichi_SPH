import taichi as ti
import numpy as np
from ..basic_op.type import *
from ..basic_op.vec_op import *

OUT_OF_RANGE = -1

@ti.dataclass
class Cell_struct:
    part_num: ti.i32
    buffer_start_index: ti.i32

@ti.dataclass
class Part_struct:
    cell_id: ti.i32
    push_seq: ti.i32
    push_buffer: ti.i32

@ti.data_oriented
class Neighb_search_s:
    def __init__(
            self,
            dim: ti.template(),
            cell_size: ti.template(),
            lb: ti.template(),
            rt: ti.template(),
            part_num: ti.template(),
            stack_top: ti.template(),
            pos: ti.template(),
    ):
        # 赋值并检测参数知否合规
        self.dim = dim
        self.cell_size = cell_size
        self.lb = lb
        self.rt = rt
        self.part_num = part_num
        self.stack_top = stack_top
        self.pos = pos
        self.parameter_cehck()

        # 计算网格数量(cell_num_vec, cell_num) 并准备好网格编码器(cell_coder)
        self.cell_num = val_i()
        self.cell_num_vec = vecx_i(self.dim[None])
        self.cell_coder = vecx_i(self.dim[None])
        self.calculate_cell_param()
        
        self.part_num_in_cell = ti.field(ti.i32, (self.cell_num[None]))
        self.IDbuffer_start_index_of_cell = ti.field(ti.i32, (self.cell_num[None]))

        self.cell_id_of_part = ti.field(ti.i32, (part_num[None]))
        self.push_seq_of_part = ti.field(ti.i32, (part_num[None]))
        self.IDbuffer_of_part = ti.field(ti.i32, (part_num[None]))
        
        
        self.push_index_timer = val_i(0)
    
    def parameter_cehck(self):
        if self.dim[None] != 2 and self.dim[None] != 3:
            raise Exception("dim must be 2 or 3")
        if self.cell_size[None] <= 0:
            raise Exception("cell_size must be positive")
        if self.lb[None].n != self.rt[None].n:
            raise Exception("lb and rt must have the same shape")
        if self.lb[None].n != self.dim[None]:
            raise Exception("lb and rt must have the same shape as dim")
        for i in range(self.dim[None]):
            if self.lb[None][i] >= self.rt[None][i]:
                raise Exception("lb must be smaller than rt")

    # 计算网格数量并准备好网格编码器
    @ti.kernel
    def calculate_cell_param(self):
        dim = ti.static(self.cell_coder[None].n)
        assign(self.cell_num_vec[None], ti.ceil(
            (self.rt[None] - self.lb[None]) / self.cell_size[None]).cast(ti.i32))
        for i in ti.static(range(dim)):
            self.cell_coder[None][i] = 1
        self.cell_num[None] = 1
        for i in ti.static(range(dim)):
            self.cell_coder[None][i] = self.cell_num[None]
            self.cell_num[None] *= int(self.cell_num_vec[None][i])
    
    @ti.func
    def encode_into_cell(
        self,
        part_pos: ti.template(),  # 粒子的位置
    ):
        return int((part_pos - self.lb[None]) // self.cell_size[None]).dot(self.cell_coder[None])

    @ti.kernel
    def update_part_in_cell(
            self,
    ):
        for cell_id in range(self.cell_num[None]):
            self.part_num_in_cell[cell_id] = 0
        
        for part_id in range(self.stack_top[None]):
            cell_id = self.encode_into_cell(self.pos[part_id])
            if 0 < cell_id < self.cell_num[None]:
                self.cell_id_of_part[part_id] = cell_id
                self.push_seq_of_part[part_id] = ti.atomic_add(self.part_num_in_cell[cell_id], 1)
            else:
                self.push_seq_of_part[part_id] = OUT_OF_RANGE
        
        timer = 0
        for cell_id in range(self.cell_num[None]):
            if self.part_num_in_cell[cell_id] > 0:
                self.IDbuffer_start_index_of_cell[cell_id] = ti.atomic_add(timer, self.part_num_in_cell[cell_id])
        
        for part_id in range(self.stack_top[None]):
            cell_id = self.cell_id_of_part[part_id]
            if self.push_seq_of_part[part_id] != OUT_OF_RANGE:
                index = self.IDbuffer_start_index_of_cell[cell_id] + self.push_seq_of_part[part_id]
                self.IDbuffer_of_part[index] = part_id
    
    @ti.kernel
    def get_part_in_cell(
            self,
    ):
        sum = 0
        for cell_id in range(self.cell_num[None]):
            if self.part_num_in_cell[cell_id] > 0:
                ti.atomic_add(sum, self.part_num_in_cell[cell_id])
                print("There are ", self.part_num_in_cell[cell_id], " particles in cell ", cell_id)
        print("There are ", sum, " particles in total")



    @ti.func
    def loop_neighbors(self, part_pos:ti.template(), range:ti.template(), task:ti.template(), ret:ti.template()):
        # part_pos: 2/3 dim vector, float
        # range: float value
        # task: function
        # ret: return value, a specific array
        cell_id = self.encode_into_cell(part_pos)

