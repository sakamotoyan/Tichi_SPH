# runs WCSPH

import re
import taichi as ti
import ti_sph as tsph
import numpy as np
from plyfile import PlyData, PlyElement
from ti_sph.func_util import clean_attr_arr, clean_attr_val, clean_attr_mat
from ti_sph.solver.WCSPH import WCSPH
from ti_sph.data_struct.SDF_boundary import boxContainer
import math

ti.init(arch=ti.cuda)

# /// --- CONNFIG --- ///
config_capacity = ["info_space", "info_discretization", "info_sim", "info_gui"]
config = tsph.Config(dim=3, capacity_list=config_capacity)

#initialize each value in config
# /// space ///
config_space = ti.static(config.space)
config_space.dim[None] = 3
config_space.lb[None] = [-8, -8, -8]
config_space.rt[None] = [8, 8, 8]

# /// discretization ///
config_discre = ti.static(config.discre)
config_discre.part_size[None] = 0.1 # 0.06
config_discre.cs[None] = 100
config_discre.cfl_factor[None] = 0.5
config_discre.dt[None] = (
    tsph.fixed_dt(
        config_discre.cs[None],
        config_discre.part_size[None],
        config_discre.cfl_factor[None],
    )
)
standart_dt = config_discre.dt[None]
config_discre.inv_dt[None] = 1 / config_discre.dt[None]

# /// sim ///
config_sim = ti.static(config.sim)
config_sim.gravity[None] = ti.Vector([0, -9.8, 0])
config_sim.kinematic_vis[None] = 1e-3

# /// gui ///
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
# /// --- END OF CONNFIG --- ///

# initialize Neighb_Cell object
# /// --- NEIGHB --- ///
config_neighb = tsph.Neighb_Cell(
    dim=3,
    struct_space=config_space,
    cell_size=config_discre.part_size[None] * 2,
    search_range=1,
)
# /// --- END OF NEIGHB --- ///

# initialize Node objects (fluid and boundary) and add particles
"""""" """ OBJECT """ """"""
# /// --- INIT OBJECT --- ///
# /// FLUID ///
fluid_capacity = [
    "node_basic",
    "node_color",
    "node_sph",
    "node_wcsph",
    "node_neighb_search",
]
fluid = tsph.Node(
    dim=config_space.dim[None],
    id=0,
    node_num=int(1e5),
    capacity_list=fluid_capacity,
)
fluid_node_num = fluid.push_cube_with_basic_attr(
    lb=ti.Vector([-1, -1, 0]),
    rt=ti.Vector([1, 1, 1]),
    span=config_discre.part_size[None],
    size=config_discre.part_size[None],
    rest_density=1000,
    color=ti.Vector([0, 1, 1]),
)

print("pushed fluid parts: " + str(fluid_node_num))

# /// BOUND ///
bound = boxContainer(
    min_corner=ti.Vector([-1.5, -1.5, -1.5]), 
    max_corner=ti.Vector([1.5, 1.5, 1.5])
)

print("created bound sdf.")

# /// --- END OF INIT OBJECT --- ///

search_template = tsph.Neighb_search_template(
    dim=config_space.dim[None],
    search_range=1,
)

fluid_neighb_grid = tsph.Neighb_grid(
    obj=fluid,
    dim=config_space.dim[None],
    lb=config_space.lb,
    rt=config_space.rt,
    cell_size=config_discre.part_size[None] * 2,
)

fluid_neighb_grid.register(obj_pos=fluid.basic.pos)
print('basic.pos',fluid.basic.pos.to_numpy()[0:fluid_node_num])

# /// --- INIT SOLVER --- ///
# /// assign solver ///
fluid_wc_solver = WCSPH(
    obj=fluid,
    dt=config_discre.dt[None],
    background_neighb_grid=fluid_neighb_grid,
    search_template=search_template,
    cs=config_discre.cs[None],
    # port_sph_psi="wcsph.sph_compression_ratio",
    # port_rest_psi="wcsph.one",
    # port_X="basic.rest_volume",
)


# /// --- END OF INIT SOLVER --- ///

# define simulation loop
# /// --- LOOP --- ///
def loop():
    # /// dynamic dt ///
    tsph.cfl_dt(
        obj=fluid,
        obj_size=fluid.basic.size,
        obj_vel=fluid.basic.vel,
        cfl_factor=config_discre.cfl_factor,
        standard_dt=standart_dt,
        output_dt=config_discre.dt,
        output_inv_dt=config_discre.inv_dt,
    )

    fluid_wc_solver.update_dt(config_discre.dt[None])

    # /// neighb search ///
    fluid_neighb_grid.register(obj_pos=fluid.basic.pos)

    # /// compute density ///
    fluid_wc_solver.clear_psi()
    fluid_wc_solver.compute_psi_from(fluid_wc_solver)
    # from sdf boundary
    fluid_wc_solver.compute_psi_from_sdf(
        from_sdf=bound,
    )

    # /// copy vel to vel_adv ///
    fluid_wc_solver.set_vel_adv()

    # /// acc to vel_adv ///
    fluid_wc_solver.clear_acc()
    fluid_wc_solver.add_acc(config_sim.gravity)
    fluid_wc_solver.add_acc_from_vis(
        kinetic_vis_coeff=config_sim.kinematic_vis,
        from_solver=fluid_wc_solver,
    )
    fluid_wc_solver.update_vel_adv_from_acc()

    # /// compute pressure and pressure vel_adv ///
    fluid_wc_solver.compute_pressure()
    fluid_wc_solver.update_vel_adv_from_pressure(
        from_solver=fluid_wc_solver,
    )
    fluid_wc_solver.update_vel_adv_from_sdf(
        from_sdf=bound,
    )
    # fluid_wc_solver.simpleBoundary()

    # /// fluid: vel_adv to vel (set vel = vel_adv) ///
    fluid.attr_set_arr(
        obj_attr=fluid_wc_solver.obj_vel,
        val_arr=fluid_wc_solver.obj_vel_adv,
    )

    # /// fluid: vel to pos ///
    fluid_wc_solver.time_integral_arr(
        obj_frac=fluid_wc_solver.obj_vel,
        obj_output_int=fluid_wc_solver.obj_pos,
    )


# /// --- END OF LOOP --- ///

loop()
print('obj_sph_psi', fluid_wc_solver.obj_sph_psi.to_numpy()[0:fluid_node_num])
print('obj_rest_psi', fluid_wc_solver.obj_rest_psi.to_numpy()[0:fluid_node_num])
print('obj_pressure', fluid_wc_solver.obj_pressure.to_numpy()[0:fluid_node_num])
print('obj_acc', fluid_wc_solver.obj_acc.to_numpy()[0:fluid_node_num])
print('obj_vel_adv', fluid_wc_solver.obj_vel_adv.to_numpy()[0:fluid_node_num])
print('obj_vel', fluid_wc_solver.obj_vel.to_numpy()[0:fluid_node_num])
print('obj_X', fluid_wc_solver.obj_X.to_numpy()[0:fluid_node_num])
print('basic.rest_volume', fluid.basic.rest_volume.to_numpy()[0:fluid_node_num])
""" GUI """
gui = tsph.Gui(config_gui)
gui.env_set_up()
loop_count = 0
while gui.window.running:
    if gui.op_system_run:
        loop()
        loop_count += 1
        print(loop_count)
    gui.monitor_listen()
    if gui.op_refresh_window:
        gui.scene_setup()
        gui.scene_add_parts(fluid, size=config_discre.part_size[None])
        # if gui.show_bound:
        #     gui.scene_add_parts(bound, size=config_discre.part_size[None])
        gui.scene_render()
