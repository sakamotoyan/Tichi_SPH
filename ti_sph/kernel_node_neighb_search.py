import taichi as ti

''' Compute neighb.cell_num & neighb.cell_num_vec & neighb.cell_coder ''' 
@ti.kernel
def calculate_neighb_cell_param(struct_neighb: ti.template(), struct_space: ti.template()):
    dim = ti.static(struct_neighb.cell_coder[None].n)
    
    struct_space = ti.static(struct_space)
    struct_neighb = ti.static(struct_neighb)
    
    struct_neighb.cell_num_vec[None] = ti.ceil(
        (struct_space.rt[None] - struct_space.lb[None]) / struct_neighb.cell_size[None])
    
    for i in ti.static(range(dim)):
        struct_neighb.cell_coder[None][i] = 1
    struct_neighb.cell_num[None] = 1
    
    for i in ti.static(range(struct_neighb.cell_coder[None].n)):
        struct_neighb.cell_coder[None][i] = struct_neighb.cell_num[None]
        struct_neighb.cell_num[None] *= int(
            struct_neighb.cell_num_vec[None][i])

        
