import taichi as ti


@ti.data_oriented
class Neighb_Cell:
    def __init__(
        self,
        dim,
        struct_space,
        cell_size,
        search_range,
    ):
        self.cell_size = ti.field(ti.f32, ())
        self.cell_num = ti.field(ti.i32, ())
        self.cell_num_vec = ti.Vector.field(dim, ti.i32, ())
        self.cell_coder = ti.Vector.field(dim, ti.i32, ())
        self.search_range = ti.field(ti.i32, ())

        self.cell_size[None] = cell_size
        self.search_range[None] = search_range

        self.search_template = ti.Vector.field(
            dim, ti.i32, (self.search_range[None] * 2 + 1) ** dim
        )
        self.neighb_dice = ti.field(ti.i32, self.search_range[None] * 2 + 1)

        self.calculate_neighb_cell_param(struct_space)

        for i in range(self.neighb_dice.shape[0]):
            self.neighb_dice[i] = -search_range + i

        for i in range(self.search_template.shape[0]):
            tmp = i
            for j in ti.static(range(dim)):
                digit = tmp // (self.neighb_dice.shape[0] ** (dim - j - 1))
                tmp = tmp % (self.neighb_dice.shape[0] ** (dim - j - 1))
                self.search_template[i][dim - j - 1] = self.neighb_dice[digit]

    @ti.kernel
    def calculate_neighb_cell_param(self, struct_space: ti.template()):
        # struct_space = ti.static(struct_space)
        struct_neighb = self

        struct_neighb.cell_num_vec[None] = ti.ceil(
            (struct_space.rt[None] - struct_space.lb[None])
            / struct_neighb.cell_size[None]
        )

        for i in ti.static(range(struct_neighb.cell_coder[None].n)):
            struct_neighb.cell_coder[None][i] = 1
        struct_neighb.cell_num[None] = 1

        for i in ti.static(range(struct_neighb.cell_coder[None].n)):
            struct_neighb.cell_coder[None][i] = struct_neighb.cell_num[None]
            struct_neighb.cell_num[None] *= int(struct_neighb.cell_num_vec[None][i])


@ti.data_oriented
class Neighb_grid:
    def __init__(
        self,
        obj,
        dim,
        lb,
        rt,
        cell_size,
    ):
        self.cell_size = ti.field(ti.f32, ())
        self.cell_num = ti.field(ti.i32, ())
        self.cell_num_vec = ti.Vector.field(dim, ti.i32, ())
        self.cell_coder = ti.Vector.field(dim, ti.i32, ())
        self.lb = ti.Vector.field(dim, ti.f32, ())
        self.rt = ti.Vector.field(dim, ti.f32, ())

        self.vec = ti.Vector.field(dim, ti.i32, (obj.info.node_num[None]))
        self.coded = ti.field(ti.i32, (obj.info.node_num[None]))
        self.sequence = ti.field(ti.i32, (obj.info.node_num[None]))
        self.part_log = ti.field(ti.i32, (obj.info.node_num[None]))

        self.obj = obj
        self.lb[None] = lb
        self.rt[None] = rt
        self.cell_size[None] = cell_size

        # self.search_template = ti.Vector.field(
        #     dim, ti.i32, (self.search_range[None] * 2 + 1) ** dim
        # )
        # self.neighb_dice = ti.field(ti.i32, self.search_range[None] * 2 + 1)
        # for i in range(self.neighb_dice.shape[0]):
        #     self.neighb_dice[i] = -search_range + i

        # for i in range(self.search_template.shape[0]):
        #     tmp = i
        #     for j in ti.static(range(dim)):
        #         digit = tmp // (self.neighb_dice.shape[0] ** (dim - j - 1))
        #         tmp = tmp % (self.neighb_dice.shape[0] ** (dim - j - 1))
        #         self.search_template[i][dim - j - 1] = self.neighb_dice[digit]

        self.calculate_neighb_cell_param()

        self.part_count = ti.field(ti.i32, (self.cell_num[None]))
        self.part_shift = ti.field(ti.i32, (self.cell_num[None]))

    @ti.kernel
    def calculate_neighb_cell_param(self):
        self.cell_num_vec[None] = ti.ceil(
            (self.rt[None] - self.lb[None]) / self.cell_size[None]
        ).cast(ti.i32)

        for i in ti.static(range(self.cell_coder[None].n)):
            self.cell_coder[None][i] = 1
        self.cell_num[None] = 1

        for i in ti.static(range(self.cell_coder[None].n)):
            self.cell_coder[None][i] = self.cell_num[None]
            self.cell_num[None] *= int(self.cell_num_vec[None][i])

    @ti.func
    def node_encode(
        self,
        node_pos: ti.template(),
        lb: ti.template(),
        cell_size: ti.template(),
    ):
        return int((node_pos - lb[None]) // cell_size[None])

    @ti.kernel
    def register(
        self,
        obj_pos: ti.template(),
    ):
        for i in range(self.cell_num[None]):
            self.part_count[i] = 0
            self.sequence[i] = -1

        for i in range(self.obj.info.stack_top[None]):
            self.vec[i] = self.node_encode(
                obj_pos[i],
                self.lb,
                self.cell_size,
            )
            self.coded[i] = self.vec[i].dot(self.cell_coder[None])
            if 0 < self.coded[i] < self.cell_num[None]:
                self.sequence[i] = ti.atomic_add(self.part_count[self.coded[i]], 1)
        sum = 0
        for i in range(self.cell_num[None]):
            self.part_shift[i] = ti.atomic_add(sum, self.part_count[i])
        for i in range(self.obj.info.stack_top[None]):
            if not self.sequence[i] < 0:
                seq = self.part_shift[self.coded[i]] + self.sequence[i]
                self.part_log[seq] = i

    @ti.func
    def get_located_cell(self, pos):
        cell_vec = self.node_encode(
            pos,
            self.lb,
            self.cell_size,
        )
        return cell_vec

    @ti.func
    def get_neighb_cell_index(self, located_cell, cell_iter, neighb_search_template):
        cell_coded = (
            located_cell + neighb_search_template.search_template[cell_iter]
        ).dot(self.cell_coder[None])
        return cell_coded
    
    @ti.func
    def within_grid(self, cell_index):
        return (0 < cell_index < self.cell_num[None])
    
    @ti.func
    def get_cell_part_num(self, cell_index):
        return self.part_count[cell_index]
    
    @ti.func
    def get_neighb_part_id(self, cell_index, neighb_part_index):
        shift = self.part_shift[cell_index] + neighb_part_index
        return self.part_log[shift]

    @ti.func
    def get_nid_list(
        self,
        pos,
        search_template,
    ):
        nid_list = []
        cell_vec = self.node_encode(
            pos,
            self.lb,
            self.cell_size,
        )
        for cell_tpl in ti.static(range(search_template.search_template.shape[0])):
            cell_coded = (cell_vec + search_template.search_template[cell_tpl]).dot(
                self.cell_coder[None]
            )
            if 0 < cell_coded < self.cell_num[None]:
                for j in range(self.part_count[cell_coded]):
                    shift = self.part_shift[cell_coded] + j
                    nid = self.part_log[shift]
                    nid_list.append(nid)
        return nid_list


@ti.data_oriented
class Neighb_search_template:
    def __init__(
        self,
        dim,
        search_range,
    ):
        self.search_range = ti.field(ti.i32, ())
        self.search_range[None] = search_range

        self.search_template = ti.Vector.field(
            dim, ti.i32, (self.search_range[None] * 2 + 1) ** dim
        )

        self.neighb_dice = ti.field(ti.i32, self.search_range[None] * 2 + 1)

        for i in range(self.neighb_dice.shape[0]):
            self.neighb_dice[i] = -search_range + i

        for i in range(self.search_template.shape[0]):
            tmp = i
            for j in ti.static(range(dim)):
                digit = tmp // (self.neighb_dice.shape[0] ** (dim - j - 1))
                tmp = tmp % (self.neighb_dice.shape[0] ** (dim - j - 1))
                self.search_template[i][dim - j - 1] = self.neighb_dice[digit]

    @ti.func
    def get_neighb_cell_num(self):
        return self.search_template.shape[0]

@ti.data_oriented
class Neighb_grid_slim:
    def __init__(
        self,
        part_num,
        dim,
        lb,
        rt,
        cell_size,
    ):
        self.cell_size = ti.field(ti.f32, ())
        self.cell_num = ti.field(ti.i32, ())
        self.cell_num_vec = ti.Vector.field(dim, ti.i32, ())
        self.cell_coder = ti.Vector.field(dim, ti.i32, ())
        self.lb = ti.Vector.field(dim, ti.f32, ())
        self.rt = ti.Vector.field(dim, ti.f32, ())

        self.vec = ti.Vector.field(dim, ti.i32, (part_num))
        self.coded = ti.field(ti.i32, (part_num))
        self.sequence = ti.field(ti.i32, (part_num))
        self.part_log = ti.field(ti.i32, (part_num))

        self.lb[None] = lb
        self.rt[None] = rt
        self.cell_size[None] = cell_size

        # self.search_template = ti.Vector.field(
        #     dim, ti.i32, (self.search_range[None] * 2 + 1) ** dim
        # )
        # self.neighb_dice = ti.field(ti.i32, self.search_range[None] * 2 + 1)
        # for i in range(self.neighb_dice.shape[0]):
        #     self.neighb_dice[i] = -search_range + i

        # for i in range(self.search_template.shape[0]):
        #     tmp = i
        #     for j in ti.static(range(dim)):
        #         digit = tmp // (self.neighb_dice.shape[0] ** (dim - j - 1))
        #         tmp = tmp % (self.neighb_dice.shape[0] ** (dim - j - 1))
        #         self.search_template[i][dim - j - 1] = self.neighb_dice[digit]

        self.calculate_neighb_cell_param()

        self.part_count = ti.field(ti.i32, (self.cell_num[None]))
        self.part_shift = ti.field(ti.i32, (self.cell_num[None]))

    @ti.kernel
    def calculate_neighb_cell_param(self):
        self.cell_num_vec[None] = ti.ceil(
            (self.rt[None] - self.lb[None]) / self.cell_size[None]
        ).cast(ti.i32)

        for i in ti.static(range(self.cell_coder[None].n)):
            self.cell_coder[None][i] = 1
        self.cell_num[None] = 1

        for i in ti.static(range(self.cell_coder[None].n)):
            self.cell_coder[None][i] = self.cell_num[None]
            self.cell_num[None] *= int(self.cell_num_vec[None][i])

    @ti.func
    def node_encode(
        self,
        node_pos: ti.template(),
        lb: ti.template(),
        cell_size: ti.template(),
    ):
        return int((node_pos - lb[None]) // cell_size[None])

    @ti.kernel
    def register(
        self,
        stack_top: ti.template(),
        obj_pos: ti.template(),
    ):
        for i in range(self.cell_num[None]):
            self.part_count[i] = 0
        
        for i in range(stack_top[None]):
            self.sequence[i] = -1

        for i in range(stack_top[None]):
            self.vec[i] = self.node_encode(
                obj_pos[i],
                self.lb,
                self.cell_size,
            )
            self.coded[i] = self.vec[i].dot(self.cell_coder[None])
            if 0 < self.coded[i] < self.cell_num[None]:
                self.sequence[i] = ti.atomic_add(self.part_count[self.coded[i]], 1)
        sum = 0
        for i in range(self.cell_num[None]):
            self.part_shift[i] = ti.atomic_add(sum, self.part_count[i])
        for i in range(stack_top[None]):
            if not self.sequence[i] < 0:
                seq = self.part_shift[self.coded[i]] + self.sequence[i]
                self.part_log[seq] = i

    @ti.func
    def get_located_cell(self, pos):
        cell_vec = self.node_encode(
            pos,
            self.lb,
            self.cell_size,
        )
        return cell_vec

    @ti.func
    def get_neighb_cell_index(self, located_cell, cell_iter, neighb_search_template):
        cell_coded = (
            located_cell + neighb_search_template.search_template[cell_iter]
        ).dot(self.cell_coder[None])
        return cell_coded
    
    @ti.func
    def within_grid(self, cell_index):
        return (0 < cell_index < self.cell_num[None])
    
    @ti.func
    def get_cell_part_num(self, cell_index):
        return self.part_count[cell_index]
    
    @ti.func
    def get_neighb_part_id(self, cell_index, neighb_part_index):
        shift = self.part_shift[cell_index] + neighb_part_index
        return self.part_log[shift]

    @ti.func
    def get_nid_list(
        self,
        pos,
        search_template,
    ):
        nid_list = []
        cell_vec = self.node_encode(
            pos,
            self.lb,
            self.cell_size,
        )
        for cell_tpl in ti.static(range(search_template.search_template.shape[0])):
            cell_coded = (cell_vec + search_template.search_template[cell_tpl]).dot(
                self.cell_coder[None]
            )
            if 0 < cell_coded < self.cell_num[None]:
                for j in range(self.part_count[cell_coded]):
                    shift = self.part_shift[cell_coded] + j
                    nid = self.part_log[shift]
                    nid_list.append(nid)
        return nid_list


@ti.data_oriented
class Neighb_search_template:
    def __init__(
        self,
        dim,
        search_range,
    ):
        self.search_range = ti.field(ti.i32, ())
        self.search_range[None] = search_range

        self.search_template = ti.Vector.field(
            dim, ti.i32, (self.search_range[None] * 2 + 1) ** dim
        )

        self.neighb_dice = ti.field(ti.i32, self.search_range[None] * 2 + 1)

        for i in range(self.neighb_dice.shape[0]):
            self.neighb_dice[i] = -search_range + i

        for i in range(self.search_template.shape[0]):
            tmp = i
            for j in ti.static(range(dim)):
                digit = tmp // (self.neighb_dice.shape[0] ** (dim - j - 1))
                tmp = tmp % (self.neighb_dice.shape[0] ** (dim - j - 1))
                self.search_template[i][dim - j - 1] = self.neighb_dice[digit]

    @ti.func
    def get_neighb_cell_num(self):
        return self.search_template.shape[0]
