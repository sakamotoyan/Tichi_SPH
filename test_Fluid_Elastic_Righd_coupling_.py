from cgitb import reset
import taichi as ti
import ti_sph as tsph
import numpy as np
from plyfile import PlyData, PlyElement
from ti_sph.func_util import clean_attr_arr, clean_attr_val, clean_attr_mat
from ti_sph.solver import DFSPH_layer
from ti_sph.solver.ISPH_Elastic import ISPH_Elastic
from ti_sph.solver.DFSPH import DFSPH
import math

from ti_sph.solver.SPH_kernel import cfl_dt

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

# sim
config_sim = ti.static(config.sim)
config_sim.gravity[None] = ti.Vector([0, -9.8, 0])
config_sim.kinematic_vis[None] = 1e-1

# discretization
config_discre = ti.static(config.discre)
config_discre.part_size[None] = 0.075
config_discre.cs[None] = 220
config_discre.cfl_factor[None] = 0.5
config_discre.dt[None] = (
    tsph.fixed_dt(
        config_discre.cs[None],
        config_discre.part_size[None],
        config_discre.cfl_factor[None],
    )
    * 3
)
config_discre.inv_dt[None] = 1 / config_discre.dt[None]
standart_dt = config_discre.dt[None]

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

"""""" """ OBJECT """ """"""
elastic_list = []
elastic_neighb_list = []
elastic_neighb_0_list = []
elastic_solver_list = []
elastic_df_solver_list = []
# ELASTIC
elastic_capacity = [
    "node_basic",
    "node_color",
    "node_sph",
    "node_ISPH_Elastic",
    "node_implicit_sph",
]
elastic_list.append(
    tsph.Node(
        dim=config_space.dim[None],
        id=0,
        node_num=int(1e5),
        capacity_list=elastic_capacity,
    )
)
elastic_node_num = elastic_list[0].push_cube_with_basic_attr(
    lb=ti.Vector([-1, -0.2, -1]),
    rt=ti.Vector([1, 0.1, 1]),
    span=config_discre.part_size[None],
    size=config_discre.part_size[None],
    rest_density=500,
    color=ti.Vector([1, 0, 1]),
)

elastic_list.append(
    tsph.Node(
        dim=config_space.dim[None],
        id=0,
        node_num=int(1e5),
        capacity_list=elastic_capacity,
    )
)
elastic_node_num = elastic_list[1].push_cube_with_basic_attr(
    lb=ti.Vector([-0.5, 0.4, -0.5]),
    rt=ti.Vector([0.5, 1, 0.5]),
    span=config_discre.part_size[None],
    size=config_discre.part_size[None],
    rest_density=100,
    color=ti.Vector([0.8, 0, 0.8]),
)

# FLUID
fluid_capacity = [
    "node_basic",
    "node_color",
    "node_sph",
    "node_implicit_sph",
]
fluid = tsph.Node(
    dim=config_space.dim[None],
    id=0,
    node_num=int(1e6),
    capacity_list=fluid_capacity,
)
fluid_node_num = fluid.push_cube_with_basic_attr(
    lb=ti.Vector([-1, -1.4, -2.5]),
    rt=ti.Vector([1, -0.4, 2.5]),
    span=config_discre.part_size[None],
    size=config_discre.part_size[None],
    rest_density=1000,
    color=ti.Vector([0, 1, 1]),
)
devi = ti.Vector([0, 1.2, 0])
fluid_node_num = fluid.push_cube_with_basic_attr(
    lb=ti.Vector([-1, -1.4, -2.5]) + devi,
    rt=ti.Vector([1, -0.4, -1.5]) + devi,
    span=config_discre.part_size[None],
    size=config_discre.part_size[None],
    rest_density=5000,
    color=ti.Vector([0, 1, 0.5]),
)

# BOUND
bound_capacity = [
    "node_basic",
    "node_color",
    "node_sph",
    "node_implicit_sph",
]
bound = tsph.Node(
    dim=config_space.dim[None],
    id=0,
    node_num=int(1e6),
    capacity_list=bound_capacity,
)
bound_node_num = bound.push_box_with_basic_attr(
    lb=ti.Vector([-1.5, -1.5, -3]),
    rt=ti.Vector([1.5, 5, 3]),
    span=config_discre.part_size[None],
    size=config_discre.part_size[None],
    layers=2,
    rest_density=1000,
    color=ti.Vector([0.3, 0.3, 0.3]),
)


"""""" """ COMPUTE """ """"""
# /// --- NEIGHB --- ///
search_template = tsph.Neighb_search_template(
    dim=config_space.dim[None],
    search_range=1,
)

for elastic in elastic_list:
    elastic_neighb_list.append(
        tsph.Neighb_grid(
            obj=elastic,
            dim=config_space.dim[None],
            lb=config_space.lb,
            rt=config_space.rt,
            cell_size=config_discre.part_size[None] * 2,
        )
    )
    elastic_neighb_0_list.append(
        tsph.Neighb_grid(
            obj=elastic,
            dim=config_space.dim[None],
            lb=config_space.lb,
            rt=config_space.rt,
            cell_size=config_discre.part_size[None] * 2,
        )
    )
bound_neighb_grid = tsph.Neighb_grid(
    obj=bound,
    dim=config_space.dim[None],
    lb=config_space.lb,
    rt=config_space.rt,
    cell_size=config_discre.part_size[None] * 2,
)
fluid_neighb_grid = tsph.Neighb_grid(
    obj=fluid,
    dim=config_space.dim[None],
    lb=config_space.lb,
    rt=config_space.rt,
    cell_size=config_discre.part_size[None] * 2,
)

# /// --- INIT SOLVER --- ///
# /// ISPH_Elastic ///
for elastic, neighb, neighb_0 in zip(
    elastic_list, elastic_neighb_list, elastic_neighb_0_list
):
    elastic_solver_list.append(
        ISPH_Elastic(
            obj=elastic,
            dt=config_discre.dt[None],
            background_neighb_grid=neighb,
            background_neighb_grid_0=neighb_0,
            search_template=search_template,
        )
    )

elastic_solver_list[0].K[None]=1e5
elastic_solver_list[0].G[None]=5e3
elastic_solver_list[1].K[None]=1e5
elastic_solver_list[1].G[None]=1e4

fluid_df_solver = DFSPH(
    obj=fluid,
    dt=config_discre.dt[None],
    background_neighb_grid=fluid_neighb_grid,
    search_template=search_template,
    # port_sph_psi="implicit_sph.sph_compression_ratio",
    # port_rest_psi="implicit_sph.one",
    # port_X="basic.rest_volume",
)

for elastic, neighb in zip(elastic_list, elastic_neighb_list):
    elastic_df_solver_list.append(
        DFSPH(
            obj=elastic,
            dt=config_discre.dt[None],
            background_neighb_grid=neighb,
            search_template=search_template,
            # port_sph_psi="implicit_sph.sph_compression_ratio",
            # port_rest_psi="implicit_sph.one",
            # port_X="basic.rest_volume",
        )
    )

bound_df_solver = DFSPH(
    obj=bound,
    dt=config_discre.dt[None],
    background_neighb_grid=bound_neighb_grid,
    search_template=search_template,
    # port_sph_psi="implicit_sph.sph_compression_ratio",
    # port_rest_psi="implicit_sph.one",
    # port_X="basic.rest_volume",
)

solvers = (
    [bound_df_solver, fluid_df_solver] + elastic_df_solver_list + elastic_solver_list
)

coupling_solver = [bound_df_solver, fluid_df_solver] + elastic_df_solver_list
solver_type = ["static", "fluid"]
for i in range(len(elastic_df_solver_list)):
    solver_type.append("elastic")

df_solver_layer = DFSPH_layer(coupling_solver, solver_type, number_density=True)


# /// --- LOOP --- ///
def loop():

    tsph.cfl_dt(
        obj=fluid,
        obj_size=fluid.basic.size,
        obj_vel=fluid.basic.vel,
        cfl_factor=config_discre.cfl_factor,
        standard_dt=standart_dt,
        output_dt=config_discre.dt,
        output_inv_dt=config_discre.inv_dt,
    )

    for solver in solvers:
        solver.update_dt(config_discre.dt[None])

    #  /// neighb search ///
    for elastic, neighb in zip(elastic_list, elastic_neighb_list):
        neighb.register(obj_pos=elastic.basic.pos)
    bound_neighb_grid.register(obj_pos=bound.basic.pos)
    fluid_neighb_grid.register(obj_pos=fluid.basic.pos)

    #  /// elastic sim  ///
    for elastic, elastic_solver in zip(elastic_list, elastic_solver_list):
        elastic.clear(elastic.basic.force)
        elastic.clear(elastic.basic.acc)

        #  /// advection  ///
        elastic_solver.compute_vis(
            kinetic_vis_coeff=config_sim.kinematic_vis,
            output_acc=elastic.basic.acc,
        )
        elastic.attr_add(
            obj_attr=elastic.basic.acc,
            val=config_sim.gravity,
        )

        elastic_solver.time_integral_arr(
            obj_frac=elastic.basic.acc,
            obj_output_int=elastic.basic.vel,
        )

        elastic.clear(elastic.basic.force)
        elastic.clear(elastic.basic.acc)

        elastic_solver.internal_loop(output_force=elastic.basic.force)    

        elastic_solver.update_acc(
            input_force=elastic.basic.force,
            output_acc=elastic.basic.acc,
        )

        elastic_solver.time_integral_arr(
            obj_frac=elastic.basic.acc,
            obj_output_int=elastic.basic.vel,
        )

    fluid_df_solver.clear_acc()
    fluid_df_solver.add_acc(config_sim.gravity)
    fluid_df_solver.update_vel_from_acc()

    #  /// df sim  ///
    df_solver_layer.loop()

    # /// update vel to pos ///
    for elastic, elastic_solver in zip(elastic_list, elastic_solver_list):
        elastic_solver.time_integral_arr(
            obj_frac=elastic.basic.vel,
            obj_output_int=elastic.basic.pos,
        )
    fluid_df_solver.time_integral_arr(
        obj_frac=fluid_df_solver.obj_vel,
        obj_output_int=fluid_df_solver.obj_pos,
    )


# /// --- END OF LOOP --- ///
loop()
# /// --- GUI --- ///
gui = tsph.Gui(config_gui)
gui.env_set_up()
while gui.window.running:
    gui.monitor_listen()
    if gui.op_system_run == True:
        loop()
    if gui.op_refresh_window:
        gui.scene_setup()
        for elastic in elastic_list:
            gui.scene_add_parts(elastic, size=config_discre.part_size[None])
        if gui.show_bound:
            gui.scene_add_parts(fluid, size=config_discre.part_size[None])
            # gui.scene_add_parts(bound, size=config_discre.part_size[None])
        gui.scene_render()
