import taichi as ti
import numpy as np
from ..basic_op.type import *
from ..basic_op.vec_op import *

OUT_OF_RANGE = -1

@ti.data_oriented
class Neighb_cell_simple:
    def __init__(
            self,
            obj__: ti.template(), # Particle class
            cell_size_: ti.template(), # val_f()
            lb_: ti.template(), # vecx_f(dim)
            rt_: ti.template(), # vecx_f(dim)
            part_num_: ti.template(), # vali_()
            stack_top_: ti.template(), # vali_()
            pos_: ti.template(), # ti.Vector.field(dim, ti.f32, part_num)
    ):
        # 赋值并检测参数知否合规
        self.obj_ = obj__
        self.dim = ti.static(pos_.n)
        self.cell_size_ = cell_size_
        self.lb_ = lb_
        self.rt_ = rt_
        self.part_num_ = part_num_
        self.stack_top_ = stack_top_
        self.pos_ = pos_
        self.parameter_cehck()

        # 计算网格数量(cell_num_vec, cell_num) 并准备好网格编码器(cell_coder)
        self.cell_num_ = val_i()
        self.cell_num_vec_ = vecx_i(self.dim)
        self.cell_coder_ = vecx_i(self.dim)
        self.calculate_cell_param()
        
        self.part_num_in_cell_ = ti.field(ti.i32, (self.cell_num_[None]))
        self.cell_begin_pointer_ = ti.field(ti.i32, (self.cell_num_[None]))

        self.cell_id_of_part_ = ti.field(ti.i32, (part_num_[None]))
        self.part_pointer_shift_ = ti.field(ti.i32, (part_num_[None])) # 用于计算part在cell中的偏移
        self.part_id_container_ = ti.field(ti.i32, (part_num_[None]))
        
        
        self.push_index_timer_ = val_i(0)
    
    def parameter_cehck(self):
        if self.dim != 2 and self.dim != 3:
            raise Exception("dim must be 2 or 3")
        if self.cell_size_[None] <= 0:
            raise Exception("cell_size must be positive")
        if self.lb_[None].n != self.rt_[None].n:
            raise Exception("lb and rt must have the same shape")
        if self.lb_.n != self.dim:
            raise Exception("lb and rt must have the same shape as dim")
        for i in range(self.dim):
            if self.lb_[None][i] >= self.rt_[None][i]:
                raise Exception("lb must be smaller than rt")

    # 计算网格数量并准备好网格编码器
    @ti.kernel
    def calculate_cell_param(self):
        dim = ti.static(self.cell_coder_[None].n)
        assign(self.cell_num_vec_[None], ti.ceil(
            (self.rt_[None] - self.lb_[None]) / self.cell_size_[None]).cast(ti.i32))
        for i in ti.static(range(dim)):
            self.cell_coder_[None][i] = 1
        self.cell_num_[None] = 1
        for i in ti.static(range(dim)):
            self.cell_coder_[None][i] = self.cell_num_[None]
            self.cell_num_[None] *= int(self.cell_num_vec_[None][i])
    
    @ti.func
    def encode_into_cell(
        self,
        part_pos: ti.template(),  # 粒子的位置
    ):
        return ti.floor((part_pos - self.lb_[None]) // self.cell_size_[None]).cast(ti.i32).dot(self.cell_coder_[None])

    @ti.func
    def compute_cell_vec(
        self,
        part_pos: ti.template(),  # 粒子的位置
    ):
        return ti.floor((part_pos - self.lb_[None]) // self.cell_size_[None]).cast(ti.i32)
    
    @ti.func
    def encode_cell_vec(
        self,
        cell_vec: ti.template(),
    ):
        return cell_vec.dot(self.cell_coder_[None])
    
    @ti.func
    def within_cell(
        self,
        cell_vec: ti.template(),
    ):
        ans = True
        for i in ti.static(range(self.dim)):
            ans = ans and (0 <= cell_vec[i] < self.cell_num_vec_[None][i])   
        return ans
    
    @ti.func
    def get_part_num_in_cell(
        self,
        cell_id: ti.template(),
    ):
        return self.part_num_in_cell_[cell_id]
    
    @ti.func
    def get_part_id_in_cell(
        self,
        cell_id: ti.template(),
        part_shift: ti.template(),
    ):
        return self.part_id_container_[self.cell_begin_pointer_[cell_id] + part_shift]

    @ti.kernel
    def update_part_in_cell(
            self,
    ):
        for cell_id in range(self.cell_num_[None]):
            self.part_num_in_cell_[cell_id] = 0 # 清空 网格粒子数量计数器
        
        for part_id in range(self.stack_top_[None]):
            cell_id = self.encode_into_cell(self.pos_[part_id]) # 计算粒子所在网格(编码后)
            if 0 < cell_id < self.cell_num_[None]: # 如果粒子在网格范围内则 
                self.cell_id_of_part_[part_id] = cell_id # 记录粒子所在网格
                self.part_pointer_shift_[part_id] = ti.atomic_add(self.part_num_in_cell_[cell_id], 1) # 网格粒子数量计数器 + 1
            else: # 如果粒子不在网格范围内则标记为出界
                self.part_pointer_shift_[part_id] = OUT_OF_RANGE
        
        timer = 0
        for cell_id in range(self.cell_num_[None]):
            if self.part_num_in_cell_[cell_id] > 0: # 如果网格内有粒子则 设置赋予初始指针
                self.cell_begin_pointer_[cell_id] = ti.atomic_add(timer, self.part_num_in_cell_[cell_id])
        
        for part_id in range(self.stack_top_[None]):
            cell_id = self.cell_id_of_part_[part_id]
            if self.part_pointer_shift_[part_id] != OUT_OF_RANGE:
                part_pointer = self.cell_begin_pointer_[cell_id] + self.part_pointer_shift_[part_id]
                self.part_id_container_[part_pointer] = part_id
    




    # DEBUG FUNCTION
    @ti.kernel
    def get_part_in_cell(
            self,
    ):
        sum = 0
        for cell_id in range(self.cell_num_[None]):
            if self.part_num_in_cell_[cell_id] > 0:
                ti.atomic_add(sum, self.part_num_in_cell_[cell_id])
                print("There are ", self.part_num_in_cell_[cell_id], " particles in cell ", cell_id)
        print("There are ", sum, " particles in total")



    # @ti.func
    # def loop_neighbors(self, part_pos:ti.template(), range:ti.template(), task:ti.template(), ret:ti.template()):
    #     # part_pos: 2/3 dim vector, float
    #     # range: float value
    #     # task: function
    #     # ret: return value, a specific array
    #     cell_id = self.encode_into_cell(part_pos)

