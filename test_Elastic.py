import taichi as ti
import ti_sph as tsph
import numpy as np
from plyfile import PlyData, PlyElement
from ti_sph.func_util import clean_attr_arr, clean_attr_val, clean_attr_mat
from ti_sph.sim.ISPH_Elastic import ISPH_Elastic
import math

ti.init(arch=ti.cuda)


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

"""SPH_kernel"""
# sph_kernel = SPH_kernel()
# sph_kernel.set_h(obj=fluid, obj_h=fluid.sph.h, h=config_discre.part_size[None] * 2)
# sph_kernel.compute_sig(obj=fluid, obj_sig=fluid.sph.sig)
"""ISPH_Elastic"""
fluid_solver = ISPH_Elastic(obj=fluid)
fluid_solver.compute_kernel(
    obj=fluid,
    h=config_discre.part_size[None] * 2,
    obj_output_h=fluid.sph.h,
    obj_output_sig=fluid.sph.sig,
    obj_output_sig_inv_h=fluid.sph.sig_inv_h,
)

"""neighb search"""
fluid.neighb_search(config_neighb, config_space)
bound.neighb_search(config_neighb, config_space)
"""some first-time computation"""
fluid.resize(fluid.basic.pos, ti.Vector([1, 1, 1.2]))
"""compute L"""
clean_attr_mat(fluid, fluid.elastic_sph.L)
fluid_solver.compute_L(
    obj_sph=fluid.sph,
    obj_volume=fluid.basic.rest_volume,
    obj_pos_0=fluid.elastic_sph.pos_0,
    obj_output_L=fluid.elastic_sph.L,
    config_neighb=config_neighb,
    obj_located_cell=fluid.located_cell,
    neighb_cell=fluid.cell,
)


def loop():
    """compute compression"""
    clean_attr_arr(fluid, fluid.basic.acc)
    """compute F"""
    clean_attr_mat(fluid, fluid.elastic_sph.F)
    fluid_solver.compute_F(
        obj_sph=fluid.sph,
        obj_volume=fluid.basic.rest_volume,
        obj_pos_0=fluid.elastic_sph.pos_0,
        obj_pos_now=fluid.basic.pos,
        obj_L=fluid.elastic_sph.L,
        obj_output_F=fluid.elastic_sph.F,
        config_neighb=config_neighb,
        obj_located_cell=fluid.located_cell,
        neighb_cell=fluid.cell,
    )
    """compute R"""
    fluid_solver.compute_R_pd(
        obj_F=fluid.elastic_sph.F,
        obj_output_R=fluid.elastic_sph.R,
    )
    """compute F_star"""
    clean_attr_mat(fluid, fluid.elastic_sph.F)
    fluid_solver.compute_F_star(
        obj_sph=fluid.sph,
        obj_volume=fluid.basic.rest_volume,
        obj_pos_0=fluid.elastic_sph.pos_0,
        obj_pos_now=fluid.basic.pos,
        obj_R=fluid.elastic_sph.R,
        obj_L=fluid.elastic_sph.L,
        obj_output_F_star=fluid.elastic_sph.F,
        config_neighb=config_neighb,
        obj_located_cell=fluid.located_cell,
        neighb_cell=fluid.cell,
    )
    """compute epsilon"""
    fluid_solver.compute_eps(
        obj_F=fluid.elastic_sph.F,
        obj_output_eps=fluid.elastic_sph.eps,
    )
    """compute P"""
    fluid_solver.compute_P(
        obj_eps=fluid.elastic_sph.eps,
        obj_output_P=fluid.elastic_sph.P,
    )
    """compute force"""
    clean_attr_arr(fluid, fluid.elastic_sph.force)
    fluid_solver.compute_force(
        obj_sph=fluid.sph,
        obj_volume=fluid.basic.rest_volume,
        obj_pos_0=fluid.elastic_sph.pos_0,
        obj_R=fluid.elastic_sph.R,
        obj_L=fluid.elastic_sph.L,
        obj_P=fluid.elastic_sph.P,
        obj_output_force=fluid.elastic_sph.force,
        config_neighb=config_neighb,
        obj_located_cell=fluid.located_cell,
        neighb_cell=fluid.cell,
    )
    """update acc"""
    fluid.update_acc(
        obj_force=fluid.elastic_sph.force,
    )
    fluid_solver.compute_Laplacian(
        obj=fluid,
        obj_pos=fluid.basic.pos,
        nobj=fluid,
        nobj_pos=fluid.basic.pos,
        nobj_volume=fluid.basic.rest_volume,
        obj_input_attr=fluid.basic.vel,
        nobj_input_attr=fluid.basic.vel,
        coeff=1,
        obj_output_attr=fluid.basic.acc,
        config_neighb=config_neighb,
    )
    # update vel
    fluid.update_vel(dt=config_discre.dt[None])
    # update pos
    fluid.update_pos(dt=config_discre.dt[None])
    # debug
    # result = fluid.basic.vel.to_numpy()[:10]
    # np.set_printoptions(threshold=1e6)
    # print(result)


# path = tsph.trim_path_dir(".\\data\\result")
# np.savetxt(path, result)

# window = ti.ui.Window("Fluid Simulation", (640, 480))
# GUI
gui = tsph.Gui(config_gui)
gui.env_set_up()

while gui.window.running:
    gui.monitor_listen()
    if gui.op_system_run == True:
        loop()
    if gui.op_refresh_window:
        gui.scene_setup()
        gui.scene_add_parts(fluid, size=config_discre.part_size[None])
        # gui.scene_add_parts(bound, size=config_discre.part_size[None])
        gui.scene_render()


# file_seq = 0
# obj_name = 'fluid'
# path = tsph.trim_path_dir(".\\data\\")
# file_name = 'pos'

# save_data = fluid.basic.pos.to_numpy()[:fluid.info.stack_top[None]]
# pos_dtype = [('x','f4'),('y','f4'),('z','f4')]
# save_data = np.array([tuple(item) for item in save_data],dtype=pos_dtype)
# el = PlyElement.describe(save_data, 'vertex')
# # save_data.dtype = pos_dtype
# PlyData([el]).write(save_path+'pos_data.ply')
# np.save(save_path+'pos_data', save_data)

# print(save_data['x'])

# print(config.space)
# print(config.discre)
# print(config.neighb)
# print(config.sim)
# print(config.gui)


# def inverse_mat(
#     mat,
#     inv_mat,
# ):
#     mat_np = mat.to_numpy()
#     mat_inv = np.linalg.pinv(mat_np)
#     inv_mat.from_numpy(mat_inv)


# A = ti.Matrix.field(3, 3, ti.f32, (2,))
# B = ti.Matrix.field(3, 3, ti.f32, (2,))
# A[0] = [[1, 0, 0], [0, 3, 0], [0, 0, 2]]
# A[1] = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
# inverse_mat(A, B)
# print(B)


# a = ti.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# b = ti.Matrix.field(3, 3, ti.f32, (2,))


# @ti.kernel
# def pd(mat: ti.template()):
#     print(ti.Vector.one(dt=ti.f32, n=4))


# A = ti.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# pd(A)
# print(A)
