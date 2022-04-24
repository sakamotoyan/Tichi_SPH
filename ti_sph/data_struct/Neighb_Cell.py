import taichi as ti

@ti.data_oriented
class Neighb_Cell:
    def __init__(self, dim, struct_space, cell_size, search_range):
        self.cell_size = ti.field(ti.f32, ())
        self.cell_num = ti.field(ti.i32, ())
        self.cell_num_vec = ti.Vector.field(dim, ti.i32, ())
        self.cell_coder = ti.Vector.field(dim, ti.i32, ())
        self.search_range = ti.field(ti.i32, ())

        self.cell_size[None] = cell_size
        self.search_range[None] = search_range

        self.search_template = ti.Vector.field(dim, ti.i32, (self.search_range[None] * 2 + 1) ** dim)
        self.neighb_dice = ti.field(ti.i32, self.search_range[None] * 2 + 1)

        self.calculate_neighb_cell_param(struct_space)

        for i in range(self.neighb_dice.shape[0]):
            self.neighb_dice[i] = - search_range + i

        for i in range(self.search_template.shape[0]):
            tmp = i
            for j in ti.static(range(dim)):
                digit = tmp // (self.neighb_dice.shape[0] ** (dim - j - 1))
                tmp = tmp % (self.neighb_dice.shape[0] ** (dim - j - 1))
                self.search_template[i][dim - j - 1] = self.neighb_dice[digit]
        
        
    @ti.kernel
    def calculate_neighb_cell_param(self, struct_space: ti.template()):
        struct_space = ti.static(struct_space)
        struct_neighb = self

        struct_neighb.cell_num_vec[None] = ti.ceil(
            (struct_space.rt[None] - struct_space.lb[None]) / struct_neighb.cell_size[None])

        for i in ti.static(range(struct_neighb.cell_coder[None].n)):
            struct_neighb.cell_coder[None][i] = 1
        struct_neighb.cell_num[None] = 1

        for i in ti.static(range(struct_neighb.cell_coder[None].n)):
            struct_neighb.cell_coder[None][i] = struct_neighb.cell_num[None]
            struct_neighb.cell_num[None] *= int(
                struct_neighb.cell_num_vec[None][i])
