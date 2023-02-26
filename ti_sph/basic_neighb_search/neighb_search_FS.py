import taichi as ti
import numpy as np
from ..basic_op.type import *
from ..basic_op.vec_op import *

# 用于邻居搜索的粒子结构


@ti.dataclass
class part_neighb_search:
    cell_id: ti.i32  # 粒子所在的网格编号
    push_seq: ti.i32  # 粒子在网格中的推入顺序
    prev_part_id: ti.i32  # 粒子在网格中的前一个粒子编号，NO_PREV_PART 表示没有前一个粒子
    next_part_id: ti.i32  # 粒子在网格中的后一个粒子编号，NO_NEXT_PART 表示没有后一个粒子
    flag: ti.i32  # 粒子是否离开原来所在网格(CHANGED/NOT_CHANGED)，以及是否出网格边界(OUT_OF_RANGE)、未初始化(UNINITIALIZED)
    push_buffer: ti.i32  # 推入粒子的缓冲区
# 强制状态_出界: cell_id == OUT_OF_CELL, prev_part_id == NO_PREV_PART, next_part_id == NO_NEXT_PART, flag == OUT_OF_RANGE
# 强制状态_未初始化: cell_id == OUT_OF_CELL, prev_part_id == NO_PREV_PART, next_part_id == NO_NEXT_PART, flag == UNINITIALIZED
# 强制状态_网格内唯一粒子: cell_id != OUT_OF_CELL, prev_part_id = NO_PREV_PART, next_part_id = NO_NEXT_PART, flag == NOT_CHANGED or CHANGED
# 强制状态_网格内第一个粒子(且有后续粒子): cell_id != OUT_OF_CELL, prev_part_id = NO_PREV_PART, next_part_id != NO_NEXT_PART, flag == NOT_CHANGED or CHANGED
# 强制状态_网格内最后一个粒子(且有前续粒子): cell_id != OUT_OF_CELL, prev_part_id != NO_PREV_PART, next_part_id = NO_NEXT_PART, flag == NOT_CHANGED or CHANGED


# part_neighb_search中各属性取值含义
# cell_id的取值: -1表示粒子不在网格中，即粒子已经出网格边界或未初始化
OUT_OF_CELL = -1
# prev_part_id,next_part_id的取值: -1表示没有前一个粒子或没有后一个粒子
NO_PREV_PART = -1
NO_NEXT_PART = -1
# flag的取值: 意味着粒子是否离开原来所在网格，以及是否出网格边界、未初始化
NOT_CHANGED = 0
CHANGED = -1
OUT_OF_RANGE = -2
UNINITIALIZED = -3

# 用于邻居搜索的网格结构


@ti.dataclass
class cell_neighb_search:
    first_part_id: ti.i32
    last_part_id: ti.i32
    visit_num: ti.i32
    push_num: ti.i32
    push_start_index: ti.i32
    
# 强制状态_空网格: first_part_id==CELL_EMPTY_ID, last_part_id==CELL_EMPTY_ID, visit_num==CELL_EMPTY_NUM
# 强制状态_初始化: first_part_id==CELL_EMPTY_ID, last_part_id==CELL_EMPTY_ID, visit_num==CELL_EMPTY_NUM, atomic_visit_begin==0, atomic_visit_end==0


# cell_neighb_search 中各属性取值含义
# first_part_id,last_part_id的取值: 表示网格中第一个粒子的编号和最后一个粒子的编号
CELL_EMPTY_ID = -1  # 当网格中没有粒子时，first_part_id,last_part_id的取值同为 -1
# visit_num的取值: 表示网格中当前粒子数量
CELL_EMPTY_NUM = 0  # 当网格中没有粒子时，visit_num的取值为 0
# atomic_visit_begin,atomic_visit_end的取值: 为了保证粒子并行写同一网格时的原子性, 在每回邻居搜索前，将atomic_visit_begin,atomic_visit_end的值都设置为0
# step 1 当一线程a要调整网格c内数据时，先将 atomic_visit_begin 通过ti.atomic_add()加1，并获取+1前atomic_visit_begin值，记为a1
# step 2 判断 a1 是否等于 atomic_visit_end，
#   step 2 case1 如果等于，说明没有其他线程在调整网格c内数据，线程a可以调整网格c内数据
#   step 2 case2 如果不等于，说明有其他线程在调整网格c内数据，线程a需要等待，直到其他线程调整完网格c内数据
# step 3 线程a调整网格c内数据
# step 4 线程a将 atomic_visit_end 通过ti.atomic_add()加1, 表示线程a调整网格c内数据完毕, 其他等待线程可以结束step2 case2的等待, 开始step2 case1的判断, 进行step 3


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
        # 赋值并检测参数知否合规
        self.dim = val_i(dim[None])
        self.cell_size = cell_size
        self.lb = lb
        self.rt = rt
        self.obj = obj
        self.pos = obj_pos
        self.parameter_cehck()

        # 计算网格数量(cell_num_vec, cell_num) 并准备好网格编码器(cell_coder)
        self.cell_num = val_i()
        self.cell_num_vec = vecx_i(self.dim[None])
        self.cell_coder = vecx_i(self.dim[None])
        self.calculate_cell_param()

        # 两个container，一个用于存储网格(所构建的cell)，一个用于存储粒子(obj做对应的粒子属性)
        self.cell_ns_container = cell_neighb_search.field(
            shape=(self.cell_num[None],))
        self.part_ns_container = part_neighb_search.field(
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
        self.cell_ns_container.first_part_id.fill(CELL_EMPTY_ID)
        self.cell_ns_container.last_part_id.fill(CELL_EMPTY_ID)
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
