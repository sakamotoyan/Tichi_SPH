import taichi as ti
from .constructor import *
from .func_util import *


@ti.data_oriented
class Node:
    def __init__(self, dim, id, part_num, neighb_cell_num, capacity_list):

        self.info = self.node()
        self.info.id[None] = id
        self.info.part_num[None] = part_num
        self.info.stack_top[None] = 0
        self.info.neighb_cell_num[None] = neighb_cell_num
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

    @ti.kernel
    def push_attr_seq(self, obj_attr: ti.template(), attr_seq: ti.template(), pushed_part_num: ti.i32, begin_index: ti.i32):
        dim = ti.static(attr_seq.n)
        for i in range(pushed_part_num):
            i_p = i + begin_index
            for j in ti.static(range(dim)):
                obj_attr[i_p][j] = attr_seq[i][j]

    def push_pos_seq(self, obj, pushed_part_num, pos_seq):
        print('push ', pushed_part_num, ' particles')
        dim = pos_seq.shape[0]
        current_part_num = obj.info.stack_top[None]
        new_part_num = current_part_num + pushed_part_num
        pos_seq_ti = ti.Vector.field(dim, ti.Vector.field, pushed_part_num)
        pos_seq_ti.from_numpy(pos_seq)
        self.push_attr_seq(obj.basic.pos, pos_seq_ti, pushed_part_num,
                           current_part_num)
        obj.info.stack_top[None] = new_part_num
        return pushed_part_num

    @ti.kernel
    def push_cube(self, lb: ti.template(), rt: ti.template(), part_size: ti.template(), relaxing_factor: ti.template()) -> ti.i32:
        current_part_num = self.info.stack_top[None]
        pushed_part_seq_coder = ti.Vector([0, 0, 0])
        pushed_part_seq = int(ti.ceil((rt - lb) / part_size / relaxing_factor))
        dim = ti.static(self.basic.pos.n)
        for i in ti.static(range(dim)):
            if pushed_part_seq[i] == 0:
                pushed_part_seq[i] = 1  # at least push one
            # coder for seq
        tmp = 1
        for i in ti.static(range(dim)):
            pushed_part_seq_coder[i] = tmp
            tmp *= pushed_part_seq[i]
        # new part num
        pushed_part_num = 1
        for i in ti.static(range(dim)):
            pushed_part_num *= pushed_part_seq[i]
        new_part_num = current_part_num + pushed_part_num
        if new_part_num > self.info.part_num[None]:
            print('WARNING from push_cube(): overflow')
        # inject pos [1/2]
        for i in range(pushed_part_num):
            tmp = i
            for j in ti.static(range(dim - 1, -1, -1)):
                self.basic.pos[i +
                               current_part_num][j] = tmp // pushed_part_seq_coder[j]
                tmp = tmp % pushed_part_seq_coder[j]
        # inject pos [2/2]
        # pos seq times part size minus lb
        for i in range(pushed_part_num):
            self.basic.pos[i + current_part_num] *= part_size * relaxing_factor
            self.basic.pos[i + current_part_num] += lb
        # inject volume_frac & rest_volume & color
        for i in range(pushed_part_num):
            self.basic.rest_volume[i + current_part_num] = part_size**3
            self.basic.radius[i + current_part_num] = part_size
        # update part num
        self.info.stack_top[None] = new_part_num
        return pushed_part_num

    @ti.kernel
    def push_box(self, lb: ti.template(), rt: ti.template(), part_size: ti.template(), relaxing_factor: ti.template(), layers: ti.i32) -> ti.i32:
        dim = ti.static(self.basic.pos.n)

        current_part_num = self.info.stack_top[None]
        pushed_part_seq_coder = ti.Vector([0, 0, 0])

        pushed_part_seq = int(ti.ceil((rt - lb) / part_size / relaxing_factor))
        pushed_part_seq_offset = int(
            ti.ceil((rt - lb) / part_size / relaxing_factor))+(layers*2)

        for i in ti.static(range(dim)):
            if pushed_part_seq[i] == 0:
                pushed_part_seq[i] = 1  # at least push one

        tmp = 1
        for i in ti.static(range(dim)):
            pushed_part_seq_coder[i] = tmp
            tmp *= pushed_part_seq_offset[i]

        pushed_part_num = 1
        pushed_part_num_solid = 1
        for i in ti.static(range(dim)):
            pushed_part_num *= pushed_part_seq[i]
            pushed_part_num_solid *= pushed_part_seq_offset[i]
        pushed_part_num = pushed_part_num_solid - pushed_part_num
        new_part_num = current_part_num + pushed_part_num

        if new_part_num > self.info.part_num[None]:
            print('WARNING from push_box(): overflow')

        inc = current_part_num
        for i in range(pushed_part_num_solid):
            tmp = i
            a = rt - lb
            flag = True
            dim_check = 0
            for j in ti.static(range(dim - 1, -1, -1)):
                a[j] = tmp // pushed_part_seq_coder[j]
                tmp = tmp % pushed_part_seq_coder[j]
            if has_positive((a-1)-pushed_part_seq) or has_negative(a-layers):
                index = ti.atomic_add(inc, 1)
                for j in ti.static(range(dim - 1, -1, -1)):
                    self.basic.pos[index][j] = (
                        a[j]-layers) * part_size * relaxing_factor + lb[j]
        self.info.stack_top[None] = pushed_part_num
        return pushed_part_num

    @ti.kernel
    def neighb_search(self, config_neighb: ti.template(), config_space: ti.template()):
        for i in range(config_neighb.cell_num[None]):
            self.cell.part_count[i] = 0
            self.located_cell.sequence[i] = -1
        for i in range(self.info.stack_top[None]):
            self.located_cell.vec[i] = node_encode(
                self.basic.pos[i], config_space.lb, config_neighb.cell_size[None])
            self.located_cell.coded[i] = self.located_cell.vec[i].dot(
                config_neighb.cell_coder[None])
            if 0 < self.located_cell.coded[i] < config_neighb.cell_num[None]:
                self.located_cell.sequence[i] = ti.atomic_add(
                    self.cell.part_count[self.located_cell.coded[i]], 1)
        sum = 0
        for i in range(config_neighb.cell_num[None]):
            self.cell.part_shift[i] = ti.atomic_add(
                sum, self.cell.part_count[i])
        for i in range(self.info.stack_top[None]):
            if not self.located_cell.sequence[i] < 0:
                seq = self.cell.part_shift[self.located_cell.coded[i]]+self.located_cell.sequence[i]
                self.located_cell.part_log[seq] = i


@ti.kernel
def test(obj: ti.template(), nobj: ti.template(), config_neighb: ti.template(), i: ti.i32):
    cell_vec = ti.static(obj.located_cell.vec)
    for cell_tpl in range(config_neighb.search_template.shape[0]):
        cell_coded = (cell_vec[i] + config_neighb.search_template[cell_tpl]).dot(config_neighb.cell_coder[None])
        if 0 < cell_coded < config_neighb.cell_num[None]:
            for j in range(nobj.cell.part_count[cell_coded]):
                shift = nobj.cell.part_shift[cell_coded]+j
                nid = nobj.located_cell.part_log[shift]
                nobj.color.vec[nid] = [0, 0, 1]
                # print(nid)
            # print('------------------')
    obj.color.vec[i] = [1, 0, 0]