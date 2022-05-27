from turtle import shape
import taichi as ti

# node_construct()
# "node_basic" -> basic
def struct_node_basic(dim, node_num):
    struct_node_basic = ti.types.struct(
        pos=ti.types.vector(dim, ti.f32),
        vel=ti.types.vector(dim, ti.f32),
        acc=ti.types.vector(dim, ti.f32),
        force=ti.types.vector(dim, ti.f32),
        mass=ti.f32,
        rest_density=ti.f32,
        rest_volume=ti.f32,
        size=ti.f32,
    )
    return struct_node_basic.field(shape=(node_num,))


# node_construct()
# "node_sph" -> sph
def struct_node_sph(dim, node_num):
    struct_node_sph = ti.types.struct(
        h=ti.f32,
        sig=ti.f32,
        sig_inv_h=ti.f32,
        W=ti.f32,
        W_grad=ti.types.vector(dim, ti.f32),
        compression=ti.f32,
    )
    return struct_node_sph.field(shape=(node_num,))


# node_construct()
# "node_ISPH_Elastic" -> elastic_sph
def struct_node_elastic_sph(dim, node_num):
    struct_node_elastic_sph = ti.types.struct(
        force=ti.types.vector(dim, ti.f32),
        pos_0=ti.types.vector(dim, ti.f32),
        F=ti.types.matrix(dim, dim, ti.f32),
        L=ti.types.matrix(dim, dim, ti.f32),
        R=ti.types.matrix(dim, dim, ti.f32),
        eps=ti.types.matrix(dim, dim, ti.f32),
        P=ti.types.matrix(dim, dim, ti.f32),
    )
    return struct_node_elastic_sph.field(shape=(node_num,))


# node_construct()
# "node_implicit_sph" -> implicit_sph
def struct_node_implicit_sph(dim, node_num):
    struct_node_implicit_sph = ti.types.struct(
        alpha_1=ti.types.vector(dim, ti.f32),
        alpha_2=ti.f32,
        alpha=ti.f32,
        vel_adv=ti.types.vector(dim, ti.f32),
        acc_adv=ti.types.vector(dim, ti.f32),
        sph_compression_ratio=ti.f32,
        sph_density=ti.f32,
        delta_psi=ti.f32,
    )
    return struct_node_implicit_sph.field(shape=(node_num,))


# node_construct()
# "node_color" -> color
def struct_node_color(node_num):
    struct_node_color = ti.types.struct(
        hex=ti.i32,
        vec=ti.types.vector(3, ti.f32),
    )
    return struct_node_color.field(shape=(node_num,))


# node_construct()
# "node_neighb_search" -> located_cell
def struct_node_neighb_search(dim, node_num):
    struct_node_neighb_search = ti.types.struct(
        vec=ti.types.vector(dim, ti.i32),
        coded=ti.i32,
        sequence=ti.i32,
        part_log=ti.i32,
    )
    return struct_node_neighb_search.field(shape=(node_num,))


# node_neighb_cell_construct()
# "node_neighb_search" -> cell
def struct_node_neighb_cell(cell_num):
    struct_node_neighb_cell = ti.types.struct(
        part_count=ti.i32,
        part_shift=ti.i32,
    )
    return struct_node_neighb_cell.field(shape=(cell_num))
