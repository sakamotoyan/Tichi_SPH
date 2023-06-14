import taichi as ti
import numpy as np
from ..ti_sph.basic_op.type import *
from ..ti_sph.basic_op.vec_op import *

# 用于邻居搜索的粒子结构


@ti.dataclass
class Part_in_cell_state:
    cell_id: ti.i32       # Grid number where the particle is located
    push_seq: ti.i32      # Pushing sequence of the particle in the grid
    prev_part_id: ti.i32  # Previous particle number in the grid, NO_PREV_PART indicates no previous particle
    next_part_id: ti.i32  # Next particle number in the grid, NO_NEXT_PART indicates no next particle
    flag: ti.i32          # Whether the particle has left its original grid (CHANGED/NOT_CHANGED), gone out of the grid boundary (OUT_OF_RANGE), or uninitialized (UNINITIALIZED)
    push_buffer: ti.i32   # Buffer for pushing particles
# STATE -- out of boundary:      cell_id == OUT_OF_CELL, prev_part_id == NO_PREV_PART, next_part_id == NO_NEXT_PART, flag == OUT_OF_RANGE
# STATE -- uninitialized:        cell_id == OUT_OF_CELL, prev_part_id == NO_PREV_PART, next_part_id == NO_NEXT_PART, flag == UNINITIALIZED
# STATE -- only part in cell:    cell_id != OUT_OF_CELL, prev_part_id = NO_PREV_PART, next_part_id = NO_NEXT_PART, flag == NOT_CHANGED or CHANGED
# STATE -- first part in cell (with following part(s)):  cell_id != OUT_OF_CELL, prev_part_id = NO_PREV_PART, next_part_id != NO_NEXT_PART, flag == NOT_CHANGED or CHANGED
# STATE -- last part in cell  (with previous part(s)):   cell_id != OUT_OF_CELL, prev_part_id != NO_PREV_PART, next_part_id = NO_NEXT_PART, flag == NOT_CHANGED or CHANGED

# default attributes for Part_in_cell_state 
OUT_OF_CELL = -1 # for cell_id, meaning: outof grid or uninitialized
NO_PREV_PART = -1 # for prev_part_id, meaning: no previous particle
NO_NEXT_PART = -1 # for next_part_id, meaning: no next particle
NOT_CHANGED = 0    # for flag, meaning: not changed
CHANGED = -1       # for flag, meaning: changed
OUT_OF_RANGE = -2  # for flag, meaning: out of range
UNINITIALIZED = -3 # for flag, meaning: uninitialized

@ti.dataclass
class Cell_state:
    first_part_id: ti.i32
    last_part_id: ti.i32
    visit_num: ti.i32
    push_num: ti.i32
    push_start_index: ti.i32
    
# STATE -- empty cell:  first_part_id==CELL_EMPTY, last_part_id==CELL_EMPTY, visit_num==CELL_EMPTY_NUM
# STATE -- initialized: first_part_id==CELL_EMPTY, last_part_id==CELL_EMPTY, visit_num==CELL_EMPTY_NUM, atomic_visit_begin==0, atomic_visit_end==0

# default attributes for Cell_state
CELL_EMPTY = -1  # for first_part_id,last_part_id, meaning: empty cell
NO_VISIT = 0 # for visit_num, meaning: empty cell

@ti.data_oriented
class Neighb_search_FS:
    def __init__(
            self,
            dim: ti.template(),
            cell_size: ti.template(),
            lb: ti.template(),
            rt: ti.template(),
            obj: ti.template(),
            obj_pos: ti.template(),
    ):
        # get all the parameters
        self.dim = val_i(dim[None])
        self.cell_size = cell_size
        self.lb = lb
        self.rt = rt
        self.obj = obj
        self.pos = obj_pos
        self.parameter_cehck()

        # 计算网格数量(cell_num_vec, cell_num) 并准备好网格编码器(cell_coder)
        self.cell_num = val_i() # number of cells needed for the neighbor search grid
        self.cell_num_vec = vecx_i(self.dim[None])  # number of cells needed for the neighbor search grid in each dimension
        self.cell_coder = vecx_i(self.dim[None])
        self.calculate_cell_param()

        # 两个container，一个用于存储网格(所构建的cell)，一个用于存储粒子(obj做对应的粒子属性)
        self.cell_ns_container = Cell_state.field(
            shape=(self.cell_num[None],))
        self.part_ns_container = Part_in_cell_state.field(
            shape=(obj.part_num[None],))
        # 两个 container 初始化
        self.init_ns_container()


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

    # 对网格和粒子（结构体）进行初始化
    def init_ns_container(self):
        # 粒子的三个值都初始化为
        self.part_ns_container.cell_id.fill(OUT_OF_CELL)
        self.part_ns_container.prev_part_id.fill(NO_PREV_PART)
        self.part_ns_container.next_part_id.fill(NO_NEXT_PART)
        self.part_ns_container.flag.fill(UNINITIALIZED)
        # 初始化网格
        self.cell_ns_container.first_part_id.fill(CELL_EMPTY)
        self.cell_ns_container.last_part_id.fill(CELL_EMPTY)
        self.cell_ns_container.visit_num.fill(0)

    # 计算粒子所在的网格(返回值为int)

    @ti.func
    def encode_into_cell(
            self,
            part_pos: ti.template(),  # 粒子的位置
    ):
        return int((part_pos - self.lb[None]) // self.cell_size[None]).dot(self.cell_coder[None])


    # 基于粒子位置, 将粒子加入网格
    @ti.kernel
    def update_part_in_cell(
            self,
    ):
        part_struct = ti.static(self.part_ns_container)
        cell_struct = ti.static(self.cell_ns_container)
        push_index_timer = 0


        # ******* 遍历粒子 (属性清零) *******
        for part_id in range(self.obj.stack_top[None]):
            part_struct[part_id].push_buffer = 0
            part_struct[part_id].push_seq = 0


        # ******* 遍历网格 (属性清零) *******
        for cell_id in range(self.cell_num[None]):
            cell_struct[cell_id].push_start_index = 0
            cell_struct[cell_id].push_num = 0
        

        # ******* 遍历网格 (网格内粒子是否移动)*******
        for cell_id in range(self.cell_num[None]):

            # 空网格, 则跳过
            if cell_struct[cell_id].visit_num == 0:
                continue

            # 获取网格内粒子链表中的第一个粒子id
            part_now_id = cell_struct[cell_id].first_part_id
            # 获取网格内粒子链表中的第一个粒子的下一个粒子id
            part_next_id = part_struct[part_now_id].next_part_id
            # 获取网格内粒子链表长度
            part_in_cell_num = cell_struct[cell_id].visit_num
            
            # 遍历网格内的粒子
            for part_seq in range(part_in_cell_num): # 不能直接使用 cell_struct[cell_id].visit_num, 因循环中会改变visit_num

                # 基于粒子当前位置, 更新粒子现在所在网格id
                cell_new_id = self.encode_into_cell(self.pos[part_now_id])
                
                # 未移动到其他网格:
                if cell_new_id == cell_id:
                    part_struct[part_now_id].flag = NOT_CHANGED
                # 移动到其他网格:
                else:
                    # 若粒子没有离开所有网格范围, 则将粒子的所在网格更新为当前网格
                    if 0 < cell_new_id < self.cell_num[None]:
                        part_struct[part_now_id].cell_id = cell_new_id
                        part_struct[part_now_id].flag = CHANGED
                        part_struct[part_now_id].push_seq = ti.atomic_add(cell_struct[cell_new_id].push_num, 1)
                    # 若粒子离开所有网格范围, 则标记为离开网格, 置 强制状态_出界 状态
                    else:  
                        part_struct[part_now_id].cell_id = OUT_OF_CELL
                        part_struct[part_now_id].flag = OUT_OF_RANGE

                    # 若粒子不是第一个, 则把粒子的前驱的后继修改为粒子的后继
                    if part_struct[part_now_id].prev_part_id != NO_PREV_PART:
                        prev_part_id = part_struct[part_now_id].prev_part_id
                        part_struct[prev_part_id].next_part_id = part_next_id
                    # 若粒子是第一个, 则把网格的第一个粒子修改为粒子的后继
                    else:  
                        cell_struct[cell_id].first_part_id = part_next_id

                    # 若粒子不是最后一个, 则把粒子的后继的前驱修改为粒子的前驱
                    if part_next_id != NO_NEXT_PART:
                        part_struct[part_next_id].prev_part_id = part_struct[part_now_id].prev_part_id
                    # 若粒子是最后一个, 则把网格的最后一个粒子修改为粒子的前驱
                    else:  
                        cell_struct[cell_id].last_part_id = part_struct[part_now_id].prev_part_id
                    
                    # 重置当前粒子的前驱id和后继id
                    part_struct[part_now_id].prev_part_id = NO_PREV_PART
                    part_struct[part_now_id].next_part_id = NO_NEXT_PART

                    # 网格内粒子数减一
                    ti.atomic_sub(cell_struct[cell_id].visit_num, 1)

                # 更新当前粒子id为粒子的后继
                part_now_id = part_next_id
                part_next_id = part_struct[part_now_id].next_part_id


        # ******* 遍历粒子 (新加入粒子插入网格) *******
        for part_id in range(self.obj.stack_top[None]):
            # 如果粒子flag是UNINITIALIZED, 则先转为CHANGED/OUT_OF_RANGE状态
            if part_struct[part_id].flag == UNINITIALIZED:
                cell_new_id = self.encode_into_cell(self.pos[part_id])
                if 0 < cell_new_id < self.cell_num[None]:
                    part_struct[part_id].cell_id = cell_new_id
                    part_struct[part_id].flag = CHANGED
                    part_struct[part_id].push_seq = ti.atomic_add(cell_struct[cell_new_id].push_num, 1)
                else:
                    part_struct[part_id].cell_id = OUT_OF_CELL
                    part_struct[part_id].flag = OUT_OF_RANGE


        # ******* 遍历网格 (算buffer位置) *******
        for cell_id in range(self.cell_num[None]):
            # 无加入粒子, 跳过
            if cell_struct[cell_id].push_num == 0:
                continue
            # 有加入粒子, 计算网格buffer的起始位置 push_start_index
            cell_struct[cell_id].push_start_index = ti.atomic_add(push_index_timer, cell_struct[cell_id].push_num)


        # ******* 遍历粒子 (写buffer) *******
        for part_id in range(self.obj.stack_top[None]):
            if part_struct[part_id].flag == CHANGED:
                cell_id = part_struct[part_id].cell_id
                push_index = cell_struct[cell_id].push_start_index + part_struct[part_id].push_seq
                part_struct[push_index].push_buffer = part_id
                

        # ******* 遍历网格 (插入粒子) *******
        for cell_id in range(self.cell_num[None]):
            # 无加入粒子, 跳过
            if cell_struct[cell_id].push_num == 0:
                continue
            # 有加入粒子, 插入粒子
            for push_part_seq in range(cell_struct[cell_id].push_num):
                part_id = part_struct[cell_struct[cell_id].push_start_index + push_part_seq].push_buffer
                # 获取网格原来粒子的最后一个粒子id
                prev_part_id = cell_struct[cell_id].last_part_id
                # 若网格内没有粒子, 则把粒子设为网格的第一个粒子
                if cell_struct[cell_id].visit_num == 0:
                    cell_struct[cell_id].first_part_id = part_id
                else:  # 更新前驱粒子的后继
                    part_struct[prev_part_id].next_part_id = part_id
                # 更新粒子的前驱id
                part_struct[part_id].prev_part_id = prev_part_id
                # 把粒子加入到网格的访问链表的末尾
                cell_struct[cell_id].last_part_id = part_id

                # 网格内粒子数加一
                ti.atomic_add(cell_struct[cell_id].visit_num, 1)
                # 置粒子flag为NOT_CHANGED
                part_struct[part_id].flag = NOT_CHANGED
    
    
    @ti.kernel
    def get_part_in_cell(
            self,
    ):
        sum = 0
        for cell_id in range(self.cell_num[None]):
            if self.cell_ns_container[cell_id].visit_num > 0:
                ti.atomic_add(sum, self.cell_ns_container[cell_id].visit_num)
                # print("There are ", self.cell_ns_container[cell_id].visit_num, " particles in cell ", cell_id)
        print("There are ", sum, " particles in total")

    @ti.kernel
    def time_test(
            self,
    ):
        sum = 0
        for cell_id in range(self.cell_num[None]):
            self.cell_ns_container[cell_id].visit_num -= 1

    @ti.kernel
    def get_par_in_cell_debug(
            self,
    ):
        p_num = 0
        cell_max = 0
        for cell_id in range(self.cell_num[None]):
            if self.cell_ns_container[cell_id].visit_num > p_num:
                p_num = self.cell_ns_container[cell_id].visit_num
                cell_max = cell_id
        for i in range(self.cell_ns_container[cell_max].visit_num):
            self.debug_pos[i] = self.pos[self.get_p_id(cell_max, i)]
        # self.debug_index[None] = self.get_p_id(cell_max, 2).id

    @ti.func
    def get_neighb_cell_index(self, located_cell, cell_iter, neighb_search_template):
        cell_code = (
            located_cell + neighb_search_template.search_template[cell_iter]
        ).dot(self.cell_coder[None])
        return cell_code

    @ti.func
    def get_p_num_in_cell(self, cell_index):
        return self.cell_ns_container[cell_index].visit_num

    @ti.func
    def get_p_id(self, cell_index, p_i):
        part_now = self.part_ns_container[self.cell_ns_container[cell_index].first_part_id]
        for i in range(p_i):
            part_now = self.part_ns_container[part_now.next_part_id]
        return part_now.id

    # 封装 有效半径和网格范围
