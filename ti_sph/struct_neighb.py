import taichi as ti

''' STRUCT '''


def struct_node_neighb_search(dim, node_num):
    struct_node_neighb_search = ti.types.struct(
        located_cell_vec=ti.types.vector(dim, ti.i32),
        located_cell_coded=ti.i32,
        located_cell_sequence=ti.i32,
        cell_part_log=ti.i32,
    )
    return struct_node_neighb_search.field(shape=(node_num,))


def struct_node_neighb_cell(cell_num):
    struct_node_neighb_cell = ti.types.struct(
        part_count=ti.i32,
        part_shift=ti.i32,
    )
    return struct_node_neighb_cell.field(shape=(cell_num))
