# defines structs used in Node and Neighb_Cell

import taichi as ti

# basic particle attributes
# node_construct()
# "node_basic" -> basic
def struct_node_basic(dim, node_num):
    struct_node_basic = ti.types.struct(
        pos=ti.types.vector(dim, ti.f32),   # position
        vel=ti.types.vector(dim, ti.f32),   # velocity
        acc=ti.types.vector(dim, ti.f32),   # acceleration (not always used)
        force=ti.types.vector(dim, ti.f32), # force (not always used)
        mass=ti.f32,                        # mass
        rest_density=ti.f32,                # rest density
        rest_volume=ti.f32,                 # rest volume
        size=ti.f32,                        # diameter
    )
    return struct_node_basic.field(shape=(node_num,))


# sph
# node_construct()
# "node_sph" -> sph
def struct_node_sph(dim, node_num):
    struct_node_sph = ti.types.struct(
        h=ti.f32,                               # support radius
        sig=ti.f32,                             # smoothing kernel normalization factor
        sig_inv_h=ti.f32,                       # sig/h
        W=ti.f32,                               # not used, smoothing kernel (probably shouldn't be stored here since W_ij is different for different j)
        W_grad=ti.types.vector(dim, ti.f32),    # not used, gradient of smoothing kernel (probably shouldn't be stored here since W_ij is different for different j)
        compression=ti.f32,                     # not used currently, particle compression
    )
    return struct_node_sph.field(shape=(node_num,))


# elastic sph
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


# implicit sph (DFSPH)
# node_construct()
# "node_implicit_sph" -> implicit_sph
def struct_node_implicit_sph(dim, node_num):
    struct_node_implicit_sph = ti.types.struct(
        alpha_1=ti.types.vector(dim, ti.f32),   # DFSPH alpha 1st term (intermediate variable)
        alpha_2=ti.f32,                         # DFSPH alpha 2nd term (intermediate variable)
        alpha=ti.f32,                           # DFSPH alpha
        vel_adv=ti.types.vector(dim, ti.f32),   # advection velocity
        acc_adv=ti.types.vector(dim, ti.f32),   # advection acceleration
        sph_compression_ratio=ti.f32,           # not used
        sph_density=ti.f32,                     # sph density
        delta_psi=ti.f32,                       # delta psi (corresponds to density deviation)
    )
    return struct_node_implicit_sph.field(shape=(node_num,))


# particle color
# node_construct()
# "node_color" -> color
def struct_node_color(node_num):
    struct_node_color = ti.types.struct(
        hex=ti.i32,                             # hexadecimal color (e.g. 0xFFFFFF)
        vec=ti.types.vector(3, ti.f32),         # vector color (e.g. [0.5, 0.5, 0.5])
    )
    return struct_node_color.field(shape=(node_num,))


# neighbor search (per-particle variables)
# node_construct()
# "node_neighb_search" -> located_cell
def struct_node_neighb_search(dim, node_num):
    struct_node_neighb_search = ti.types.struct(
        vec=ti.types.vector(dim, ti.i32),       # cell index vector (n-dimensional) the particle belongs to
        coded=ti.i32,                           # cell index (1D) the particle belongs to
        sequence=ti.i32,                        # the sequence of the particle within its cell (temporary variable for neighbor search)
        part_log=ti.i32,                        # an array that stores the IDs of particles in each cell sequentially according to cell index
    )
    return struct_node_neighb_search.field(shape=(node_num,))


# neighbor search (per-cell variables)
# node_neighb_cell_construct()
# "node_neighb_search" -> cell
def struct_node_neighb_cell(cell_num):
    struct_node_neighb_cell = ti.types.struct(
        part_count=ti.i32,  # particle count in cell
        part_shift=ti.i32,  # begin index for this cell in located_cell.part_log
    )
    return struct_node_neighb_cell.field(shape=(cell_num))
