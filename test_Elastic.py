import taichi as ti
import ti_sph as tsph
import numpy as np
from plyfile import PlyData, PlyElement
from ti_sph.func_util import clean_attr_arr, clean_attr_val, clean_attr_mat
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
config_discre.part_size[None] = 0.1
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
# ELASTIC
elastic_capacity = [
    "node_basic",
    "node_color",
    "node_sph",
    "node_ISPH_Elastic",
    "node_implicit_sph",
    "node_neighb_search",
]
elastic_obj = tsph.Node(
    dim=config_space.dim[None],
    id=0,
    node_num=int(1e6),
    neighb_cell_num=config_neighb.cell_num[None],
    capacity_list=elastic_capacity,
)
elastic_node_num = elastic_obj.push_cube(
    ti.Vector([-1, -1.1, -1]),
    ti.Vector([1, 0.9, 1]),
    config_discre.part_size[None],
)
elastic_obj.attr_set_arr(
    obj_attr=elastic_obj.elastic_sph.pos_0, val_arr=elastic_obj.basic.pos
)
elastic_obj.push_attr(
    obj_attr=elastic_obj.basic.size,
    attr=config_discre.part_size[None],
    begin_index=elastic_obj.info.stack_top[None] - elastic_node_num,
    pushed_node_num=elastic_node_num,
)
elastic_obj.push_attr(
    elastic_obj.basic.rest_volume,
    config_discre.part_size[None] ** config_space.dim[None],
    elastic_obj.info.stack_top[None] - elastic_node_num,
    elastic_node_num,
)
elastic_obj.push_attr(
    elastic_obj.basic.rest_density,
    1000,
    elastic_obj.info.stack_top[None] - elastic_node_num,
    elastic_node_num,
)
elastic_obj.push_attr(
    elastic_obj.basic.mass,
    1000 * config_discre.part_size[None] ** config_space.dim[None],
    elastic_obj.info.stack_top[None] - elastic_node_num,
    elastic_node_num,
)
elastic_obj.push_attr(
    elastic_obj.color.vec,
    ti.Vector([0, 1, 1]),
    elastic_obj.info.stack_top[None] - elastic_node_num,
    elastic_node_num,
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
    node_num=int(1e6),
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
# /// --- INIT SOLVER --- ///
# /// ISPH_Elastic ///
elastic_solver = ISPH_Elastic(obj=elastic_obj, K=2e5, G=2e4)
contact_df_solver = DFSPH(obj=elastic_obj)
bound_df_solver = DFSPH(bound)
#  / compute kernel /
elastic_solver.compute_kernel(
    obj=elastic_obj,
    h=config_discre.part_size[None] * 2,
    obj_output_h=elastic_obj.sph.h,
    obj_output_sig=elastic_obj.sph.sig,
    obj_output_sig_inv_h=elastic_obj.sph.sig_inv_h,
)
bound_df_solver.compute_kernel(
    bound,
    config_discre.part_size[None] * 2,
    bound.sph.h,
    bound.sph.sig,
    bound.sph.sig_inv_h,
)
#  / neighb search /
elastic_obj.neighb_search(config_neighb, config_space)
bound.neighb_search(config_neighb, config_space)
# / resize pos /
elastic_obj.resize(elastic_obj.basic.pos, ti.Vector([1, 1, 1.2]))
# / compute L /
elastic_obj.clear(elastic_obj.elastic_sph.L)
elastic_solver.compute_L(
    obj_sph=elastic_obj.sph,
    obj_volume=elastic_obj.basic.rest_volume,
    obj_pos_0=elastic_obj.elastic_sph.pos_0,
    obj_output_L=elastic_obj.elastic_sph.L,
    config_neighb=config_neighb,
)


def elastic_sim():
    # / clear value /
    elastic_obj.clear(elastic_obj.basic.acc)
    elastic_obj.clear(elastic_obj.elastic_sph.F)
    elastic_obj.clear(elastic_obj.elastic_sph.force)
    # / compute F /
    elastic_solver.compute_F(
        obj=elastic_obj,
        obj_sph=elastic_obj.sph,
        obj_volume=elastic_obj.basic.rest_volume,
        obj_pos_0=elastic_obj.elastic_sph.pos_0,
        obj_pos_now=elastic_obj.basic.pos,
        obj_L=elastic_obj.elastic_sph.L,
        obj_output_F=elastic_obj.elastic_sph.F,
        config_neighb=config_neighb,
    )
    # / compute R /
    elastic_solver.compute_R_pd(
        obj=elastic_obj,
        obj_F=elastic_obj.elastic_sph.F,
        obj_output_R=elastic_obj.elastic_sph.R,
    )
    # / compute F_star with F cleared first /
    elastic_obj.clear(elastic_obj.elastic_sph.F)
    elastic_solver.compute_F_star(
        obj=elastic_obj,
        obj_sph=elastic_obj.sph,
        obj_volume=elastic_obj.basic.rest_volume,
        obj_pos_0=elastic_obj.elastic_sph.pos_0,
        obj_pos_now=elastic_obj.basic.pos,
        obj_R=elastic_obj.elastic_sph.R,
        obj_L=elastic_obj.elastic_sph.L,
        obj_output_F_star=elastic_obj.elastic_sph.F,
        config_neighb=config_neighb,
    )
    # / compute epsilon /
    elastic_solver.compute_eps(
        obj=elastic_obj,
        obj_F=elastic_obj.elastic_sph.F,
        obj_output_eps=elastic_obj.elastic_sph.eps,
    )
    # / compute P /
    elastic_solver.compute_P(
        obj=elastic_obj,
        obj_eps=elastic_obj.elastic_sph.eps,
        obj_output_P=elastic_obj.elastic_sph.P,
    )
    # / compute force /
    elastic_solver.compute_force(
        obj=elastic_obj,
        obj_sph=elastic_obj.sph,
        obj_volume=elastic_obj.basic.rest_volume,
        obj_pos_0=elastic_obj.elastic_sph.pos_0,
        obj_R=elastic_obj.elastic_sph.R,
        obj_L=elastic_obj.elastic_sph.L,
        obj_P=elastic_obj.elastic_sph.P,
        obj_output_force=elastic_obj.elastic_sph.force,
        config_neighb=config_neighb,
    )
    # / update acc /
    elastic_solver.update_acc(
        obj=elastic_obj,
        obj_mass=elastic_obj.basic.mass,
        obj_force=elastic_obj.elastic_sph.force,
        obj_output_acc=elastic_obj.basic.acc,
    )
    # / vis to acc /
    elastic_solver.compute_Laplacian(
        obj=elastic_obj,
        obj_pos=elastic_obj.basic.pos,
        nobj=elastic_obj,
        nobj_pos=elastic_obj.basic.pos,
        nobj_volume=elastic_obj.basic.rest_volume,
        obj_input_attr=elastic_obj.basic.vel,
        nobj_input_attr=elastic_obj.basic.vel,
        coeff=config_sim.kinematic_vis,
        obj_output_attr=elastic_obj.basic.acc,
        config_neighb=config_neighb,
    )
    # / gravity to acc /
    elastic_obj.attr_add(
        obj_attr=elastic_obj.basic.acc,
        val=config_sim.gravity,
    )
    # / update acc to vel /
    elastic_solver.time_integral(
        obj=elastic_obj,
        obj_frac=elastic_obj.basic.acc,
        dt=config_discre.dt,
        obj_output_int=elastic_obj.basic.vel,
    )

def contact_sim():
    # / elastic_obj /
    elastic_obj.clear(elastic_obj.implicit_sph.sph_density)
    elastic_obj.clear(elastic_obj.implicit_sph.alpha_1)
    elastic_obj.clear(elastic_obj.implicit_sph.alpha_2)
    elastic_obj.clear(elastic_obj.implicit_sph.acc_adv)
    contact_df_solver.comp_iter_count[None] = 0
    contact_df_solver.div_iter_count[None] = 0
    # / bound /
    bound.clear(bound.implicit_sph.sph_density)
    bound.clear(bound.implicit_sph.alpha_1)
    bound.clear(bound.implicit_sph.alpha_2)

    contact_df_solver.compute_psi(
        obj=elastic_obj,
        # obj_pos=elastic_obj.elastic_sph.pos_0,
        obj_pos=elastic_obj.elastic_sph.pos_0,
        nobj=elastic_obj,
        nobj_pos=elastic_obj.elastic_sph.pos_0,
        nobj_X=elastic_obj.basic.mass,
        obj_output_psi=elastic_obj.implicit_sph.sph_density,
        config_neighb=config_neighb,
    )
    contact_df_solver.compute_psi(
        obj=elastic_obj,
        obj_pos=elastic_obj.basic.pos,
        nobj=bound,
        nobj_pos=bound.basic.pos,
        nobj_X=bound.basic.mass,
        obj_output_psi=elastic_obj.implicit_sph.sph_density,
        config_neighb=config_neighb,
    )
    bound_df_solver.compute_psi(
        obj=bound,
        obj_pos=bound.basic.pos,
        nobj=bound,
        nobj_pos=bound.basic.pos,
        nobj_X=bound.basic.mass,
        obj_output_psi=bound.implicit_sph.sph_density,
        config_neighb=config_neighb,
    )
    bound_df_solver.compute_psi(
        obj=bound,
        obj_pos=bound.basic.pos,
        nobj=elastic_obj,
        nobj_pos=elastic_obj.basic.pos,
        nobj_X=elastic_obj.basic.mass,
        obj_output_psi=bound.implicit_sph.sph_density,
        config_neighb=config_neighb,
    )

    contact_df_solver.compute_alpha_1(
        obj=elastic_obj,
        obj_pos=elastic_obj.basic.pos,
        nobj=bound,
        nobj_pos=bound.basic.pos,
        nobj_X=bound.basic.mass,
        obj_output_alpha_1=elastic_obj.implicit_sph.alpha_1,
        config_neighb=config_neighb,
    )
    contact_df_solver.compute_alpha(
        obj=elastic_obj,
        obj_mass=elastic_obj.basic.mass,
        obj_alpha_1=elastic_obj.implicit_sph.alpha_1,
        obj_alpha_2=elastic_obj.implicit_sph.alpha_2,
        obj_output_alpha=elastic_obj.implicit_sph.alpha,
    )

    bound_df_solver.compute_alpha_2(
        obj=bound,
        obj_pos=bound.basic.pos,
        nobj=elastic_obj,
        nobj_pos=elastic_obj.basic.pos,
        nobj_mass=elastic_obj.basic.mass,
        nobj_X=elastic_obj.basic.mass,
        obj_output_alpha_2=bound.implicit_sph.alpha_2,
        config_neighb=config_neighb,
    )
    # / bound /
    bound_df_solver.compute_alpha(
        obj=bound,
        obj_mass=bound.basic.mass,
        obj_alpha_1=bound.implicit_sph.alpha_1,
        obj_alpha_2=bound.implicit_sph.alpha_2,
        obj_output_alpha=bound.implicit_sph.alpha,
    )

    elastic_obj.attr_set_arr(
        obj_attr=elastic_obj.implicit_sph.vel_adv,
        val_arr=elastic_obj.basic.vel,
    )

    while contact_df_solver.is_compressible():
        contact_df_solver.comp_iter_count[None] += 1

        contact_df_solver.compute_delta_psi(
            obj=elastic_obj,
            obj_sph_psi=elastic_obj.implicit_sph.sph_density,
            obj_rest_psi=elastic_obj.basic.rest_density,
            obj_output_delta_psi=elastic_obj.implicit_sph.delta_psi,
        )
        bound_df_solver.compute_delta_psi(
            obj=bound,
            obj_sph_psi=bound.implicit_sph.sph_density,
            obj_rest_psi=bound.basic.rest_density,
            obj_output_delta_psi=bound.implicit_sph.delta_psi,
        )

        contact_df_solver.compute_adv_psi_advection(
            obj=elastic_obj,
            obj_pos=elastic_obj.basic.pos,
            obj_vel_adv=elastic_obj.implicit_sph.vel_adv,
            nobj=bound,
            nobj_pos=bound.basic.pos,
            nobj_vel_adv=bound.implicit_sph.vel_adv,
            nobj_X=bound.basic.mass,
            dt=config_discre.dt,
            obj_output_delta_psi=elastic_obj.implicit_sph.delta_psi,
            config_neighb=config_neighb,
        )
        bound_df_solver.compute_adv_psi_advection(
            obj=bound,
            obj_pos=bound.basic.pos,
            obj_vel_adv=bound.implicit_sph.vel_adv,
            nobj=elastic_obj,
            nobj_pos=elastic_obj.basic.pos,
            nobj_vel_adv=elastic_obj.implicit_sph.vel_adv,
            nobj_X=elastic_obj.basic.mass,
            dt=config_discre.dt,
            obj_output_delta_psi=bound.implicit_sph.delta_psi,
            config_neighb=config_neighb,
        )

        contact_df_solver.statistic_non_negative_delta_psi(
            obj=elastic_obj,
            obj_rest_psi=elastic_obj.basic.rest_density,
            obj_output_delta_psi=elastic_obj.implicit_sph.delta_psi,
        )
        bound_df_solver.statistic_non_negative_delta_psi(
            obj=bound,
            obj_rest_psi=bound.basic.rest_density,
            obj_output_delta_psi=bound.implicit_sph.delta_psi,
        )

        contact_df_solver.update_vel_adv(
            obj=elastic_obj,
            obj_pos=elastic_obj.basic.pos,
            obj_X=elastic_obj.basic.mass,
            obj_delta_psi=elastic_obj.implicit_sph.delta_psi,
            obj_alpha=elastic_obj.implicit_sph.alpha,
            obj_mass=elastic_obj.basic.mass,
            nobj=bound,
            nobj_pos=bound.basic.pos,
            nobj_delta_psi=bound.implicit_sph.delta_psi,
            nobj_X=bound.basic.mass,
            nobj_alpha=bound.implicit_sph.alpha,
            inv_dt=config_discre.inv_dt,
            obj_output_vel_adv=elastic_obj.implicit_sph.vel_adv,
            config_neighb=config_neighb,
        )

    elastic_obj.attr_set_arr(
        obj_attr=elastic_obj.basic.vel,
        val_arr=elastic_obj.implicit_sph.vel_adv,
    )

# /// --- LOOP --- ///
def loop():
    #  / neighb search /
    elastic_obj.neighb_search(config_neighb, config_space)
    bound.neighb_search(config_neighb, config_space)
    #  / elastic sim  /
    elastic_sim()
    contact_sim()
    # / update vel to pos /
    elastic_solver.time_integral(
        obj=elastic_obj,
        obj_frac=elastic_obj.basic.vel,
        dt=config_discre.dt,
        obj_output_int=elastic_obj.basic.pos,
    )


# /// --- END OF LOOP --- ///

#  /// debug ///
# path = tsph.trim_path_dir(".\\data\\result")
# np.savetxt(path, result)

# /// --- GUI --- ///
gui = tsph.Gui(config_gui)
gui.env_set_up()
loop()
while gui.window.running:
    gui.monitor_listen()
    if gui.op_system_run == True:
        loop()
    if gui.op_refresh_window:
        gui.scene_setup()
        gui.scene_add_parts(elastic_obj, size=config_discre.part_size[None])
        if gui.show_bound:
            gui.scene_add_parts(bound, size=config_discre.part_size[None])
        gui.scene_render()
