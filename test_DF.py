# runs DFSPH

import re
import taichi as ti
import ti_sph as tsph
import numpy as np
from plyfile import PlyData, PlyElement
from ti_sph.func_util import clean_attr_arr, clean_attr_val, clean_attr_mat
from ti_sph.sim.DFSPH import DFSPH
import math

ti.init(arch=ti.cuda)


#initialize each value in config
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
) * 5
config_discre.inv_dt[None] = 1 / config_discre.dt[None]
# sim
config_sim = ti.static(config.sim)
config_sim.gravity[None] = ti.Vector([0, -9.8, 0])
config_sim.kinematic_vis[None] = 1e-3
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

# initialize Neighb_Cell object
config_neighb = tsph.Neighb_Cell(
    dim=3,
    struct_space=config_space,
    cell_size=config_discre.part_size[None] * 2,
    search_range=1,
)

# initialize Node objects (fluid and boundary) and add particles
"""""" """ OBJECT """ """"""
# FLUID
fluid_capacity = [
    "node_basic",
    "node_color",
    "node_sph",
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
    bound.basic.size,
    config_discre.part_size[None],
    bound.info.stack_top[None] - bound_node_num,
    bound_node_num,
)
bound.push_attr(
    bound.basic.rest_volume,
    config_discre.part_size[None] ** config_space.dim[None],
    bound.info.stack_top[None] - bound_node_num,
    bound_node_num,
)
bound.push_attr(
    bound.basic.rest_density,
    1000,
    bound.info.stack_top[None] - bound_node_num,
    bound_node_num,
)
bound.push_attr(
    bound.basic.mass,
    1000 * config_discre.part_size[None] ** config_space.dim[None],
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

# define simulation loop
def loop():
    """dynamic dt"""
    tsph.cfl_dt(
        obj=fluid,
        obj_size=fluid.basic.size,
        obj_vel=fluid.basic.vel,
        cfl_factor=config_discre.cfl_factor,
        min_acc_norm=20,
        output_dt=config_discre.dt,
        output_inv_dt=config_discre.inv_dt,
    )
    print(config_discre.dt[None])
    """neighb search"""
    # fluid
    fluid.neighb_search(config_neighb, config_space)
    # bound
    bound.neighb_search(config_neighb, config_space)

    """clear value"""
    # fluid
    fluid.clear(fluid.implicit_sph.sph_density)
    fluid.clear(fluid.implicit_sph.alpha_1)
    fluid.clear(fluid.implicit_sph.alpha_2)
    fluid.clear(fluid.implicit_sph.acc_adv)
    fluid_df_solver.comp_iter_count[None] = 0
    fluid_df_solver.div_iter_count[None] = 0
    # bound
    bound.clear(bound.implicit_sph.sph_density)
    bound.clear(bound.implicit_sph.alpha_1)
    bound.clear(bound.implicit_sph.alpha_2)

    """compute density"""
    # fluid <- fluid
    fluid_df_solver.compute_psi(
        obj=fluid,
        obj_pos=fluid.basic.pos,
        nobj=fluid,
        nobj_pos=fluid.basic.pos,
        nobj_X=fluid.basic.mass,
        obj_output_psi=fluid.implicit_sph.sph_density,
        config_neighb=config_neighb,
    )
    # fluid <- bound
    fluid_df_solver.compute_psi(
        obj=fluid,
        obj_pos=fluid.basic.pos,
        nobj=bound,
        nobj_pos=bound.basic.pos,
        nobj_X=bound.basic.mass,
        obj_output_psi=fluid.implicit_sph.sph_density,
        config_neighb=config_neighb,
    )
    # bound <- bound
    bound_df_solver.compute_psi(
        obj=bound,
        obj_pos=bound.basic.pos,
        nobj=bound,
        nobj_pos=bound.basic.pos,
        nobj_X=bound.basic.mass,
        obj_output_psi=bound.implicit_sph.sph_density,
        config_neighb=config_neighb,
    )
    # bound <- fluid
    bound_df_solver.compute_psi(
        obj=bound,
        obj_pos=bound.basic.pos,
        nobj=fluid,
        nobj_pos=fluid.basic.pos,
        nobj_X=fluid.basic.mass,
        obj_output_psi=bound.implicit_sph.sph_density,
        config_neighb=config_neighb,
    )

    """compute alpha"""
    # fluid <- fluid
    fluid_df_solver.compute_alpha_1(
        obj=fluid,
        obj_pos=fluid.basic.pos,
        nobj=fluid,
        nobj_pos=fluid.basic.pos,
        nobj_X=fluid.basic.mass,
        obj_output_alpha_1=fluid.implicit_sph.alpha_1,
        config_neighb=config_neighb,
    )
    # fluid <- bound
    fluid_df_solver.compute_alpha_1(
        obj=fluid,
        obj_pos=fluid.basic.pos,
        nobj=bound,
        nobj_pos=bound.basic.pos,
        nobj_X=bound.basic.mass,
        obj_output_alpha_1=fluid.implicit_sph.alpha_1,
        config_neighb=config_neighb,
    )
    # fluid <- fluid
    fluid_df_solver.compute_alpha_2(
        obj=fluid,
        obj_pos=fluid.basic.pos,
        nobj=fluid,
        nobj_pos=fluid.basic.pos,
        nobj_mass=fluid.basic.mass,
        nobj_X=fluid.basic.mass,
        obj_output_alpha_2=fluid.implicit_sph.alpha_2,
        config_neighb=config_neighb,
    )
    # fluid
    fluid_df_solver.compute_alpha(
        obj=fluid,
        obj_mass=fluid.basic.mass,
        obj_alpha_1=fluid.implicit_sph.alpha_1,
        obj_alpha_2=fluid.implicit_sph.alpha_2,
        obj_output_alpha=fluid.implicit_sph.alpha,
    )
    # bound <- fluid
    bound_df_solver.compute_alpha_2(
        obj=bound,
        obj_pos=bound.basic.pos,
        nobj=fluid,
        nobj_pos=fluid.basic.pos,
        nobj_mass=fluid.basic.mass,
        nobj_X=fluid.basic.mass,
        obj_output_alpha_2=bound.implicit_sph.alpha_2,
        config_neighb=config_neighb,
    )
    # bound
    bound_df_solver.compute_alpha(
        obj=bound,
        obj_mass=bound.basic.mass,
        obj_alpha_1=bound.implicit_sph.alpha_1,
        obj_alpha_2=bound.implicit_sph.alpha_2,
        obj_output_alpha=bound.implicit_sph.alpha,
    )

    # fluid
    """compute vel_adv"""
    # gravity
    fluid.attr_add(
        obj_attr=fluid.implicit_sph.acc_adv,
        val=config_sim.gravity,
    )
    # viscosity fluid inside
    fluid_df_solver.compute_Laplacian(
        obj=fluid,
        obj_pos=fluid.basic.pos,
        nobj=fluid,
        nobj_pos=fluid.basic.pos,
        nobj_volume=fluid.basic.rest_volume,
        obj_input_attr=fluid.basic.vel,
        nobj_input_attr=fluid.basic.vel,
        coeff=config_sim.kinematic_vis,
        obj_output_attr=fluid.implicit_sph.acc_adv,
        config_neighb=config_neighb,
    )
    # add velocity (set vel_adv = vel)
    fluid.attr_set_arr(
        obj_attr=fluid.implicit_sph.vel_adv,
        val_arr=fluid.basic.vel,
    )
    # update vel_adv with acc_adv
    fluid_df_solver.time_integral(
        obj=fluid,
        obj_frac=fluid.implicit_sph.acc_adv,
        dt=config_discre.dt,
        obj_output_int=fluid.implicit_sph.vel_adv,
    )

    """incompressible solver"""
    while fluid_df_solver.is_compressible():
        fluid_df_solver.comp_iter_count[None] += 1

        """compute delta density"""
        # fluid
        fluid_df_solver.compute_delta_psi(
            obj=fluid,
            obj_sph_psi=fluid.implicit_sph.sph_density,
            obj_rest_psi=fluid.basic.rest_density,
            obj_output_delta_psi=fluid.implicit_sph.delta_psi,
        )
        # bound
        bound_df_solver.compute_delta_psi(
            obj=bound,
            obj_sph_psi=bound.implicit_sph.sph_density,
            obj_rest_psi=bound.basic.rest_density,
            obj_output_delta_psi=bound.implicit_sph.delta_psi,
        )

        # fluid <- fluid
        fluid_df_solver.compute_adv_psi_advection(
            obj=fluid,
            obj_pos=fluid.basic.pos,
            obj_vel_adv=fluid.implicit_sph.vel_adv,
            nobj=fluid,
            nobj_pos=fluid.basic.pos,
            nobj_vel_adv=fluid.implicit_sph.vel_adv,
            nobj_X=fluid.basic.mass,
            dt=config_discre.dt,
            obj_output_delta_psi=fluid.implicit_sph.delta_psi,
            config_neighb=config_neighb,
        )
        # fluid <- bound
        fluid_df_solver.compute_adv_psi_advection(
            obj=fluid,
            obj_pos=fluid.basic.pos,
            obj_vel_adv=fluid.implicit_sph.vel_adv,
            nobj=bound,
            nobj_pos=bound.basic.pos,
            nobj_vel_adv=bound.implicit_sph.vel_adv,
            nobj_X=bound.basic.mass,
            dt=config_discre.dt,
            obj_output_delta_psi=fluid.implicit_sph.delta_psi,
            config_neighb=config_neighb,
        )
        # bound <- fluid
        bound_df_solver.compute_adv_psi_advection(
            obj=bound,
            obj_pos=bound.basic.pos,
            obj_vel_adv=bound.implicit_sph.vel_adv,
            nobj=fluid,
            nobj_pos=fluid.basic.pos,
            nobj_vel_adv=fluid.implicit_sph.vel_adv,
            nobj_X=fluid.basic.mass,
            dt=config_discre.dt,
            obj_output_delta_psi=bound.implicit_sph.delta_psi,
            config_neighb=config_neighb,
        )

        # fluid
        fluid_df_solver.statistic_non_negative_delta_psi(
            obj=fluid,
            obj_rest_psi=fluid.basic.rest_density,
            obj_output_delta_psi=fluid.implicit_sph.delta_psi,
        )
        # bound
        bound_df_solver.statistic_non_negative_delta_psi(
            obj=bound,
            obj_rest_psi=bound.basic.rest_density,
            obj_output_delta_psi=bound.implicit_sph.delta_psi,
        )

        # fluid <- fluid
        fluid_df_solver.update_vel_adv(
            obj=fluid,
            obj_pos=fluid.basic.pos,
            obj_X=fluid.basic.mass,
            obj_delta_psi=fluid.implicit_sph.delta_psi,
            obj_alpha=fluid.implicit_sph.alpha,
            obj_mass=fluid.basic.mass,
            nobj=fluid,
            nobj_pos=fluid.basic.pos,
            nobj_delta_psi=fluid.implicit_sph.delta_psi,
            nobj_X=fluid.basic.mass,
            nobj_alpha=fluid.implicit_sph.alpha,
            inv_dt=config_discre.inv_dt,
            obj_output_vel_adv=fluid.implicit_sph.vel_adv,
            config_neighb=config_neighb,
        )
        # fluid <- bound
        fluid_df_solver.update_vel_adv(
            obj=fluid,
            obj_pos=fluid.basic.pos,
            obj_X=fluid.basic.mass,
            obj_delta_psi=fluid.implicit_sph.delta_psi,
            obj_alpha=fluid.implicit_sph.alpha,
            obj_mass=fluid.basic.mass,
            nobj=bound,
            nobj_pos=bound.basic.pos,
            nobj_delta_psi=bound.implicit_sph.delta_psi,
            nobj_X=bound.basic.mass,
            nobj_alpha=bound.implicit_sph.alpha,
            inv_dt=config_discre.inv_dt,
            obj_output_vel_adv=fluid.implicit_sph.vel_adv,
            config_neighb=config_neighb,
        )

        # print(fluid_df_solver.comp_iter_count[None])

    # fluid: vel_adv to vel (set vel = vel_adv)
    fluid.attr_set_arr(obj_attr=fluid.basic.vel, val_arr=fluid.implicit_sph.vel_adv)
    # fluid: vel to pos (update position)
    fluid_df_solver.time_integral(
        obj=fluid,
        obj_frac=fluid.basic.vel,
        dt=config_discre.dt,
        obj_output_int=fluid.basic.pos,
    )


loop()


# result = bound.basic.mass
# print(result.to_numpy()[0:10])
# result = bound.basic.rest_volume
# print(result.to_numpy()[0:10])

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
