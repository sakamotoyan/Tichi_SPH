from turtle import shape
import taichi as ti


def struct_node_basic(dim, node_num):
    struct_node_basic = ti.types.struct(
        pos=ti.types.vector(dim, ti.f32),
        vel=ti.types.vector(dim, ti.f32),
        acc=ti.types.vector(dim, ti.f32),
        mass=ti.f32,
        rest_density=ti.f32,
        rest_volume=ti.f32,
        radius=ti.f32
    )
    return struct_node_basic.field(shape=(node_num,))


def struct_node_implicit_sph(dim, node_num):
    struct_node_implicit_sph = ti.types.struct(
        W=ti.f32,
        W_grad=ti.types.vector(dim, ti.f32),

        alpha_1=ti.types.vector(dim, ti.f32),
        alpha_2=ti.f32,

        vel_adv=ti.types.vector(dim, ti.f32),
        acce_adv=ti.types.vector(dim, ti.f32),

        approximated_compression_ratio=ti.f32,
        approximated_density=ti.f32,
        approximated_compression_ratio_adv=ti.f32,
        approximated_density_adv=ti.f32,
    )
    return struct_node_implicit_sph.field(shape=(node_num,))


def struct_node_color(node_num):
    struct_node_color = ti.types.struct(
        color_hex=ti.i32,
        color_vec=ti.types.vector(3, ti.f32),
    )
    return struct_node_color.field(shape=(node_num,))


@ti.kernel
def push_cube(obj: ti.template(), lb: ti.template(), rt: ti.template(), part_size:ti.template(), relaxing_factor:ti.template(), mask: ti.template()):
    current_part_num = obj.stack_top
    # generate seq (number of particles to push for each dimension)
    pushed_part_seq_coder = ti.Vector([0,0,0])
    pushed_part_seq = int(ti.ceil((rt - lb) / part_size / relaxing_factor))
    pushed_part_seq *= mask
    dim = ti.static(obj.basic.pos.n)
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
    if new_part_num > obj.info.part_num[None]:
        print('WARNING from push_cube(): overflow')
    # inject pos [1/2]
    for i in range(pushed_part_num):
        tmp = i
        for j in ti.static(range(dim - 1, -1, -1)):
            obj.basic.pos[i + current_part_num][j] = tmp // pushed_part_seq_coder[j]
            tmp = tmp % pushed_part_seq_coder[j]
    # inject pos [2/2]
    # pos seq times part size minus lb
    for i in range(pushed_part_num):
        obj.basic.pos[i + current_part_num] *= part_size * relaxing_factor
        obj.basic.pos[i + current_part_num] += lb
    # inject volume_frac & rest_volume & color
    for i in range(pushed_part_num):
        obj.basic.rest_volume[i + current_part_num] = part_size**3
    # update part num
    obj.info.stack_top[None] = new_part_num
    # update mass and rest_density
    # for i in range(obj.stack_top):
    #     obj.basic.rest_density[i] = obj.basic.rest_volume[i]
    #     config.phase_rest_density[None].dot(
    #         self.volume_frac[i])
    #     self.mass[i] = self.rest_density[i] * self.rest_volume[i]
