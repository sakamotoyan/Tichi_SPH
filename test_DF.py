import re
import taichi as ti
import ti_sph as tsph
import numpy as np
from plyfile import PlyData, PlyElement
from ti_sph.func_util import clean_attr_arr, clean_attr_val, clean_attr_mat
from ti_sph.sim.DFSPH import DFSPH
import math

ti.init(arch=ti.cpu)


"""""" """ CONFIG """ """"""
# CONFIG
config_capacity = ["info_space", "info_discretization", "info_sim", "info_gui"]
config = tsph.Config(dim=3, capacity_list=config_capacity)
# space
config_space = ti.static(config.space)
config_space.dim[None] = 3
config_space.lb[None] = [-8, -8, -8]
config_space.rt[None] = [8, 8, 8]
# discretization
config_discre = ti.static(config.discre)
config_discre.part_size[None] = 0.1
config_discre.cs[None] = 220
config_discre.cfl_factor[None] = 0.5
config_discre.dt[None] = tsph.fixed_dt(
    config_discre.cs[None],
    config_discre.part_size[None],
    config_discre.cfl_factor[None],
)
# gui
config_gui = ti.static(config.gui)
config_gui.res[None] = [1920, 1080]
config_gui.frame_rate[None] = 60
config_gui.cam_fov[None] = 55
config_gui.cam_pos[None] = [6.0, 1.0, 0.0]
config_gui.cam_look[None] = [0.0, 0.0, 0.0]
config_gui.canvas_color[None] = [0.2, 0.2, 0.6]
config_gui.ambient_light_color[None] = [0.7, 0.7, 0.7]
config_gui.point_light_pos[None] = [2, 1.5, -1.5]
config_gui.point_light_color[None] = [0.8, 0.8, 0.8]

# NEIGHB
config_neighb = tsph.Neighb_Cell(
    dim=3,
    struct_space=config_space,
    cell_size=config_discre.part_size[None] * 2,
    search_range=1,
)

"""""" """ OBJECT """ """"""
# FLUID
fluid_capacity = [
    "node_basic",
    "node_color",
    "node_sph",
    "node_ISPH_Elastic",
    "node_implicit_sph",
    "node_neighb_search",
]
fluid = tsph.Node(
    dim=config_space.dim[None],
    id=0,
    node_num=int(1e5),
    neighb_cell_num=config_neighb.cell_num[None],
    capacity_list=fluid_capacity,
)
fluid_node_num = fluid.push_cube(
    ti.Vector([-1, -1.1, -1]),
    ti.Vector([1, 0.9, 1]),
    config_discre.part_size[None],
)
fluid.set_attr_arr(obj_attr=fluid.elastic_sph.pos_0, val_arr=fluid.basic.pos)
fluid.push_attr(
    obj_attr=fluid.basic.size,
    attr=config_discre.part_size[None],
    begin_index=fluid.info.stack_top[None] - fluid_node_num,
    pushed_node_num=fluid_node_num,
)
fluid.push_attr(
    fluid.basic.rest_volume,
    config_discre.part_size[None] ** config_space.dim[None],
    fluid.info.stack_top[None] - fluid_node_num,
    fluid_node_num,
)
fluid.push_attr(
    fluid.basic.rest_density,
    1000,
    fluid.info.stack_top[None] - fluid_node_num,
    fluid_node_num,
)
fluid.push_attr(
    fluid.basic.mass,
    1000 * config_discre.part_size[None] ** config_space.dim[None],
    fluid.info.stack_top[None] - fluid_node_num,
    fluid_node_num,
)
fluid.push_attr(
    fluid.color.vec,
    ti.Vector([0, 1, 1]),
    fluid.info.stack_top[None] - fluid_node_num,
    fluid_node_num,
)
# BOUND
bound_capacity = [
    "node_basic",
    "node_color",
    "node_sph",
    "node_implicit_sph",
    "node_neighb_search",
]
bound = tsph.Node(
    dim=config_space.dim[None],
    id=0,
    node_num=int(1e5),
    neighb_cell_num=config_neighb.cell_num[None],
    capacity_list=bound_capacity,
)
bound_node_num = bound.push_box(
    ti.Vector([-1.5, -1.5, -1.5]),
    ti.Vector([1.5, 1, 1.5]),
    config_discre.part_size[None],
    2,
)
bound.push_attr(
    bound.basic.rest_volume,
    config_discre.part_size[None],
    bound.info.stack_top[None] - bound_node_num,
    bound_node_num,
)
bound.push_attr(
    bound.basic.size,
    config_discre.part_size[None] ** config_space.dim[None],
    bound.info.stack_top[None] - bound_node_num,
    bound_node_num,
)
bound.push_attr(
    bound.color.vec,
    ti.Vector([0.3, 0.3, 0.3]),
    bound.info.stack_top[None] - bound_node_num,
    bound_node_num,
)


"""""" """ COMPUTE """ """"""

"""assign solver"""
fluid_df_solver = DFSPH(fluid)
bound_df_solver = DFSPH(bound)

"""pre-computation"""
fluid_df_solver.compute_kernel(
    fluid,
    config_discre.part_size[None] * 2,
    fluid.sph.h,
    fluid.sph.sig,
    fluid.sph.sig_inv_h,
)
bound_df_solver.compute_kernel(
    bound,
    config_discre.part_size[None] * 2,
    bound.sph.h,
    bound.sph.sig,
    bound.sph.sig_inv_h,
)


def loop():

    """neighb search"""
    fluid.neighb_search(config_neighb, config_space)
    bound.neighb_search(config_neighb, config_space)

    """compute density"""
    fluid.clear(fluid.implicit_sph.approximated_density)
    fluid_df_solver.compute_density(
        obj=fluid,
        obj_pos=fluid.basic.pos,
        nobj=fluid,
        nobj_pos=fluid.basic.pos,
        nobj_mass=fluid.basic.mass,
        obj_output_density=fluid.implicit_sph.approximated_density,
        config_neighb=config_neighb,
    )

loop()

result = fluid.implicit_sph.approximated_density
print(result.to_numpy()[:1000])


# GUI
# gui = tsph.Gui(config_gui)
# gui.env_set_up()
# while gui.window.running:
#     # if gui.op_system_run == True:
#     loop()
#     gui.monitor_listen()
#     if gui.op_refresh_window:
#         gui.scene_setup()
#         gui.scene_add_parts(fluid, size=config_discre.part_size[None])
#         # gui.scene_add_parts(bound, size=config_discre.part_size[None])
#         gui.scene_render()
