from os import system
import numpy as np
import time

from taichi.lang.ops import atomic_min
from sph_obj import *
from fbm import *
from sph_jl21 import *


@ti.kernel
def SPH_neighbour_loop_template(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(config.neighb_search_template.shape[0]):
            node_code = dim_encode(obj.neighb_cell_structured_seq[i] + config.neighb_search_template[t], config)
            if 0 < node_code < config.node_num[None]:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code] + j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]


@ti.kernel
def SPH_clean_value(obj: ti.template(), config: ti.template()):
    obj.general_flag[None] = 1
    dim = ti.static(config.gravity.n)
    for i in range(obj.part_num[None]):
        obj.W[i] = 0
        obj.sph_compression[i] = 0
        obj.sph_density[i] = 0
        obj.alpha_2[i] = 0
        obj.flag[i] = 0
        for j in ti.static(range(dim)):
            obj.W_grad[i][j] = 0
            obj.acce[i][j] = 0
            obj.acce_adv[i][j] = 0
            obj.alpha_1[i][j] = 0
            obj.pressure_force[i][j] = 0

@ti.kernel
def cfl_condition(obj: ti.template(), config: ti.template()):
    config.dt[None] = config.part_size[1] / config.cs[None] * 5
    for i in range(obj.part_num[None]):
        v_norm = obj.vel[i].norm()
        if v_norm > 1e-4:
            atomic_min(config.dt[None], config.part_size[1] / v_norm * config.cfl_factor[None])


@ti.kernel
def SPH_prepare_attr(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(config.neighb_search_template.shape[0]):
            node_code = dim_encode(obj.neighb_cell_structured_seq[i] + config.neighb_search_template[t], config)
            if 0 < node_code < config.node_num[None]:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code] + j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        Wr = W((obj.pos[i] - nobj.pos[neighb_pid]).norm(), config)
                        obj.W[i] += Wr
                        obj.sph_compression[i] += Wr * nobj.rest_volume[neighb_pid]
                        obj.sph_density[i] += Wr * nobj.mass[neighb_pid]


@ti.kernel
def SPH_prepare_alpha_1(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(config.neighb_search_template.shape[0]):
            node_code = dim_encode(obj.neighb_cell_structured_seq[i] + config.neighb_search_template[t], config)
            if 0 < node_code < config.node_num[None]:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code] + j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        xij = obj.pos[i] - nobj.pos[neighb_pid]
                        r = xij.norm()
                        if r > 0:
                            obj.alpha_1[i] += nobj.X[neighb_pid] * xij / r * W_grad(r, config)


@ti.kernel
def SPH_prepare_alpha_2(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(config.neighb_search_template.shape[0]):
            node_code = dim_encode(obj.neighb_cell_structured_seq[i] + config.neighb_search_template[t], config)
            if 0 < node_code < config.node_num[None]:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code] + j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        r = (obj.pos[i] - nobj.pos[neighb_pid]).norm()
                        if r > 0:
                            obj.alpha_2[i] += W_grad(r, config) ** 2 * nobj.X[neighb_pid] ** 2 / nobj.mass[neighb_pid]


@ti.kernel
def SPH_prepare_alpha(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.alpha[i] = obj.alpha_1[i].dot(obj.alpha_1[i]) / obj.mass[i] + obj.alpha_2[i]
        if obj.alpha[i] < 1e-4:
            obj.alpha[i] = 1e-4


@ti.kernel
def SPH_advection_gravity_acc(obj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        obj.acce_adv[i] += config.gravity[None]


@ti.kernel
def SPH_advection_viscosity_acc(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(config.neighb_search_template.shape[0]):
            node_code = dim_encode(obj.neighb_cell_structured_seq[i] + config.neighb_search_template[t], config)
            if 0 < node_code < config.node_num[None]:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code] + j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        xij = obj.pos[i] - nobj.pos[neighb_pid]
                        r = xij.norm()
                        if r > 0:
                            obj.acce_adv[i] += W_lap(xij, r, nobj.X[neighb_pid] / nobj.sph_psi[neighb_pid],
                                                     obj.vel[i] - nobj.vel[neighb_pid], config) * config.dynamic_viscosity[None] / obj.rest_density[i]


# @ti.kernel
# def FBM_correct_vel_from_phase_vel(obj: ti.template(), config: ti.template()):
#     phase_num = ti.static(config.phase_rest_density.n)
#     dim = ti.static(config.gravity.n)
#     for i in range(obj.part_num[None]):
#         obj.vel_adv[i] *= 0
#         for k in ti.static(range(phase_num)):
#             obj.vel_adv[i] += obj.volume_frac[i][k] * obj.phase_vel[i, k]
#         for k in ti.static(range(phase_num)):
#             obj.drift_vel[i, k] = obj.phase_vel[i, k] - obj.vel_adv[i]

@ti.kernel
def FBM_update_phase_vel_from_drift_vel_and_vel_adv(obj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    dim = ti.static(config.gravity.n)
    for i in range(obj.part_num[None]):
        for k in ti.static(range(phase_num)):
            obj.phase_vel[i, k] = obj.vel_adv[i] + obj.drift_vel[i, k]


@ti.kernel
def SPH_advection_surface_tension_acc(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
    dim = ti.static(config.gravity.n)
    for i in range(obj.part_num[None]):
        for j in ti.static(range(dim)):
            obj.normal[i][j] = 0
        for t in range(config.neighb_search_template.shape[0]):
            node_code = dim_encode(
                obj.neighb_cell_structured_seq[i] + config.neighb_search_template[t], config)  # index of node to search
            if 0 < node_code < config.node_num[None]:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code] + j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        xij = obj.pos[i] - nobj.pos[neighb_pid]
                        r = xij.norm()
                        if r > 0:
                            obj.normal[i] += -nobj.X[neighb_pid] / nobj.sph_psi[neighb_pid] * W_grad(r, config) * (xij) / r
        obj.normal[i] *= config.kernel_h[1]
    for i in range(obj.part_num[None]):
        for t in range(config.neighb_search_template.shape[0]):
            node_code = dim_encode(obj.neighb_cell_structured_seq[i] + config.neighb_search_template[t], config)
            if 0 < node_code < config.node_num[None]:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code] + j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        xij = obj.pos[i] - nobj.pos[neighb_pid]
                        r = xij.norm()
                        # only phase 0 has surface tension now
                        if r > 0 and obj.volume_frac[i][0] > 0.99 and nobj.volume_frac[neighb_pid][0] > 0.99:
                            cohesion = -config.surface_tension_gamma[None] * nobj.mass[neighb_pid] * C(r, config) * xij / r
                            curvature = config.surface_tension_gamma[None] * (obj.normal[i] - nobj.normal[neighb_pid])
                            obj.acce_adv[i] += 2 * obj.rest_psi[i] / (obj.sph_psi[i] + nobj.sph_psi[neighb_pid]) * (cohesion + curvature)


@ti.kernel
def WC_pressure_val(obj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        obj.pressure[i] = (obj.rest_density[i] * config.cs[None] ** 2 / config.wc_gamma[None]) * (
                    (obj.sph_density[i] / obj.rest_density[i]) ** 7 - 1)
        if obj.pressure[i] < 0:
            obj.pressure[i] = 0


@ti.kernel
def WC_pressure_acce(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(config.neighb_search_template.shape[0]):
            node_code = dim_encode(obj.neighb_cell_structured_seq[i] + config.neighb_search_template[t], config)
            if 0 < node_code < config.node_num[None]:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code] + j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        xij = obj.pos[i] - nobj.pos[neighb_pid]
                        r = xij.norm()
                        p_term = obj.pressure[i] / ((obj.sph_density[i]) ** 2) + nobj.pressure[neighb_pid] / ((nobj.sph_density[neighb_pid]) ** 2)
                        if r > 0:
                            obj.acce_adv[i] += -p_term * nobj.mass[neighb_pid] * xij / r * W_grad(r, config)


@ti.kernel
def IPPE_adv_psi_init(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.psi_adv[i] = obj.sph_psi[i] - obj.rest_psi[i]


@ti.kernel
def IPPE_adv_psi(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(config.neighb_search_template.shape[0]):
            node_code = dim_encode(obj.neighb_cell_structured_seq[i] + config.neighb_search_template[t], config)
            if 0 < node_code < config.node_num[None]:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code] + j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        xij = obj.pos[i] - nobj.pos[neighb_pid]
                        r = xij.norm()
                        if r > 0:
                            obj.psi_adv[i] += (xij / r * W_grad(r, config)).dot(obj.vel_adv[i] - nobj.vel_adv[neighb_pid]) * nobj.X[neighb_pid] * config.dt[None]


@ti.kernel
def IPPE_psi_adv_non_negative(obj: ti.template()):
    obj.compression[None] = 0
    for i in range(obj.part_num[None]):
        if obj.psi_adv[i] < 0:
            obj.psi_adv[i] = 0
        obj.compression[None] += (obj.psi_adv[i] / obj.rest_psi[i])
    obj.compression[None] /= obj.part_num[None]

@ti.kernel
def IPPE_psi_adv_is_compressible(obj: ti.template(), config: ti.template()) -> ti.i32:
    a = 0
    for i in range(obj.part_num[None]):
        if obj.psi_adv[i] / obj.rest_psi[i] > config.divergence_threshold[None]:
            a = 1
    return a

@ti.kernel
def IPPE_update_vel_adv(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(config.neighb_search_template.shape[0]):
            node_code = dim_encode(obj.neighb_cell_structured_seq[i] + config.neighb_search_template[t], config)
            if 0 < node_code < config.node_num[None]:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code] + j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        xij = obj.pos[i] - nobj.pos[neighb_pid]
                        r = xij.norm()
                        if r > 0:
                            obj.vel_adv[i] += -(1 / config.dt[None]) * ((obj.psi_adv[i] * nobj.X[neighb_pid] / obj.alpha[i]) + (
                                        nobj.psi_adv[neighb_pid] * obj.X[i] / nobj.alpha[neighb_pid])) * (xij / r * W_grad(r, config)) / obj.mass[i]


@ti.kernel
def SPH_advection_update_vel_adv(obj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        obj.vel_adv[i] += obj.acce_adv[i] * config.dt[None]


@ti.kernel
def SPH_vel_2_vel_adv(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.vel_adv[i] = obj.vel[i]


@ti.kernel
def SPH_vel_adv_2_vel(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.vel[i] = obj.vel_adv[i]


@ti.kernel
def SPH_update_pos(obj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        obj.pos[i] += obj.vel[i] * config.dt[None]


@ti.kernel
def SPH_update_energy(obj: ti.template(), config: ti.template()):
    obj.statistics_kinetic_energy[None] = 0
    obj.statistics_gravity_potential_energy[None] = 0

    for i in range(obj.part_num[None]):
        obj.statistics_kinetic_energy[None] += 0.5 * obj.mass[i] * obj.vel[i].norm_sqr()
        obj.statistics_gravity_potential_energy[None] += -obj.mass[i] * config.gravity[None][1] * (obj.pos[i][1] - config.sim_space_lb[None][1])


@ti.kernel
def SPH_update_mass(obj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        obj.rest_density[i] = config.phase_rest_density[None].dot(obj.volume_frac[i])
        obj.mass[i] = obj.rest_density[i] * obj.rest_volume[i]


@ti.kernel
def SPH_update_color(obj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    for i in range(obj.part_num[None]):
        color = ti.Vector([0.0, 0.0, 0.0])
        for j in ti.static(range(phase_num)):
            for k in ti.static(range(3)):
                color[k] += obj.volume_frac[i][j] * config.phase_rgb[j][k]
        for j in ti.static(range(3)):
            color[j] = min(1, color[j])
        obj.color_vector[i] = color
        obj.color[i] = rgb2hex(color)


@ti.kernel
def SPH_update_color_vector(obj: ti.template()):
    for i in range(obj.part_num[None]):
        color = hex2rgb(obj.color[i])
        obj.color_vector[i] = color

@ti.kernel
def statistics_update_energy(obj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    obj.statistics_kinetic_energy[None] = 0
    obj.statistics_gravity_potential_energy[None] = 0
    for k in ti.static(range(phase_num)):
        obj.statistics_phase_kinetic_energy[None][k] = 0
    for i in range(obj.part_num[None]):
        obj.statistics_kinetic_energy[None] += 0.5 * obj.mass[i] * obj.vel[i].norm_sqr()
        obj.statistics_gravity_potential_energy[None] += -obj.mass[i] * config.gravity[None][1] * (obj.pos[i][1] - config.sim_space_lb[None][1])
        for k in ti.static(range(phase_num)):
            mass = config.phase_rest_density[None][k] * obj.rest_volume[i] * obj.volume_frac[i][k]
            obj.statistics_phase_kinetic_energy[None][k] += 0.5 * mass * (obj.phase_vel[i, k]).norm_sqr()

@ti.kernel
def statistics_update_compression(obj: ti.template(), config: ti.template()):#average of V^0_i / V_i
    obj.statistics_volume_compression[None] = 0
    for i in range(obj.part_num[None]):
        obj.statistics_volume_compression[None] += max(obj.sph_compression[i], 1.0)
    obj.statistics_volume_compression[None] /= obj.part_num[None]

@ti.kernel
def statistics_update_volume_frac(obj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    for j in ti.static(range(phase_num)):
        obj.statistics_volume_frac[None][j] = 0
    for i in range(obj.part_num[None]):
        obj.statistics_volume_frac[None] += obj.volume_frac[i]
    obj.statistics_volume_frac[None] /= obj.part_num[None]







@ti.kernel
def map_velocity(ngrid: ti.template(), grid: ti.template(), nobj: ti.template(), config: ti.template()):
    dim = ti.static(config.gravity.n)
    for I in ti.grouped(grid.vel):
        grid_pos = grid.pos[I]  # get grid pos
        nnode = node_encode(grid_pos, config)  # get grid neighb node
        for j in ti.static(range(dim)):
            grid.vel[I][j] = 0
        for t in range(config.neighb_search_template.shape[0]):
            node_code = dim_encode(nnode + config.neighb_search_template[t], config)
            if 0 < node_code < config.node_num[None]:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code] + j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        grid.vel[I] += nobj.X[neighb_pid] / nobj.sph_psi[neighb_pid] * nobj.vel[neighb_pid] * W((grid_pos - nobj.pos[neighb_pid]).norm(), config)

###################################### SPH SOLVER ############################################
def sph_step(ngrid, fluid, bound, config):
    cfl_condition(fluid, config)
    # global div_iter_count, incom_iter_count
    """ neighbour search """
    ngrid.clear_node(config)
    ngrid.encode(fluid, config)
    ngrid.encode(bound, config)
    ngrid.mem_shift(config)
    ngrid.fill_node(fluid, config)
    ngrid.fill_node(bound, config)
    """ SPH clean value """
    SPH_clean_value(fluid, config)
    SPH_clean_value(bound, config)
    FBM_clean_value(fluid, config)
    """ SPH compute W and W_grad """
    SPH_prepare_attr(ngrid, fluid, fluid, config)
    SPH_prepare_attr(ngrid, fluid, bound, config)
    SPH_prepare_attr(ngrid, bound, bound, config)
    SPH_prepare_attr(ngrid, bound, fluid, config)
    statistics_update_compression(fluid, config)
    SPH_prepare_alpha_1(ngrid, fluid, fluid, config)
    SPH_prepare_alpha_1(ngrid, fluid, bound, config)
    SPH_prepare_alpha_2(ngrid, fluid, fluid, config)
    SPH_prepare_alpha_2(ngrid, bound, fluid, config)
    SPH_prepare_alpha(fluid)
    SPH_prepare_alpha(bound)
    
    """ synchronize """
    SPH_vel_2_vel_adv(fluid)

    """ IPPE SPH divergence """
    config.div_iter_count[None] = 0
    # is_compressible = 1
    # while config.div_iter_count[None] < config.iter_threshold_min[None] or is_compressible == 1:
    #     IPPE_adv_psi_init(fluid)
    #     # IPPE_adv_psi_init(bound)
    #     IPPE_adv_psi(ngrid, fluid, fluid, config)
    #     IPPE_adv_psi(ngrid, fluid, bound, config)
    #     # IPPE_adv_psi(ngrid, bound, fluid)
    #     IPPE_psi_adv_non_negative(fluid)
    #     # IPPE_psi_adv_non_negative(bound)
    #     is_compressible = IPPE_psi_adv_is_compressible(fluid, config)
    #     IPPE_update_vel_adv(ngrid, fluid, fluid, config)
    #     IPPE_update_vel_adv(ngrid, fluid, bound, config)
    #     config.div_iter_count[None] += 1
    #     if config.div_iter_count[None] > config.iter_threshold_max[None]:
    #         break
    # SPH_vel_adv_2_vel(fluid)

    """ SPH advection """
    """ Part 1 NEW FBM procedure """
    FBM_correct_vel_from_phase_vel(fluid, config)
    SPH_vel_adv_2_vel(fluid)
    FBM_advection_M_vis(ngrid, fluid, fluid, config)
    FBM_acc_2_phase_vel(fluid, config)
    FBM_correct_vel_from_phase_vel(fluid, config)
    SPH_vel_adv_2_vel(fluid)
    """ Part 1 NEW FBM procedure """

    SPH_advection_gravity_acc(fluid, config)
    # SPH_advection_viscosity_acc(ngrid, fluid, fluid, config)
    SPH_advection_update_vel_adv(fluid, config)

    """ IPPE SPH pressure """
    config.incom_iter_count[None] = 0
    is_compressible = 1
    while config.incom_iter_count[None] < config.iter_threshold_min[None] or is_compressible == 1:
        IPPE_adv_psi_init(fluid)
        # IPPE_adv_psi_init(bound)
        IPPE_adv_psi(ngrid, fluid, fluid, config)
        IPPE_adv_psi(ngrid, fluid, bound, config)
        # IPPE_adv_psi(ngrid, bound, fluid)
        IPPE_psi_adv_non_negative(fluid)
        # IPPE_psi_adv_non_negative(bound)
        is_compressible = IPPE_psi_adv_is_compressible(fluid, config)
        IPPE_update_vel_adv(ngrid, fluid, fluid, config)
        IPPE_update_vel_adv(ngrid, fluid, bound, config)
        config.incom_iter_count[None] += 1
        if config.incom_iter_count[None] > config.iter_threshold_max[None]:
            break

    """ Part 2 NEW FBM procedure """
    FBM_convect(ngrid, fluid, fluid, config)
    while fluid.general_flag[None] > 0:
        FBM_clean_tmp(fluid, config)
        FBM_change_tmp(ngrid, fluid, fluid, config)
        # FBM_diffuse(ngrid, fluid, fluid, config)
        FBM_check_tmp(fluid)
    """ Part 2 NEW FBM procedure """

    """ SPH update """
    SPH_vel_adv_2_vel(fluid)
    FBM_update_volume_frac(fluid)
    SPH_update_mass(fluid, config)
    SPH_update_pos(fluid, config)
    SPH_update_color(fluid, config)
    statistics_update_energy(fluid, config)
    statistics_update_volume_frac(fluid, config)
    # map_velocity(ngrid, grid, fluid)

    print('phase 1:')
    print(debug_volume_frac(fluid))
    print('dt')
    print(config.dt[None])

def apply_bound_transform(bound, config):
    """ old cocktail scene """
    # bound.update_pos_part_range(config.start_id[None], config.end_id[None], config)
    # if 30 < config.time_count[None] and config.time_count[None] < config.time_down[None]:
    #     config.rod_vel.from_numpy(config.vel_down_np)
    # elif config.time_count[None]>33.5 and config.time_count[None]<38.5:
    #     ang = config.ang_spd[None] * (config.time_count[None] - config.time_down[None])
    #     config.vel_rot_np[0] = config.ang_spd[None] * config.rot_r[None] * cos(ang)
    #     config.vel_rot_np[2] = config.ang_spd[None] * config.rot_r[None] * sin(ang)
    #     config.rod_vel.from_numpy(config.vel_rot_np)
    # bound.set_vel_part_range(config.start_id[None], config.end_id[None], config.rod_vel)
    """ rewrite cocktail scene """
    # if 30 < config.time_count[None] and config.time_count[None] < config.time_down[None]:
    #     translate = config.vel_down_np * config.dt[None]
    #     bound.move_scene_obj('rod',translation_matrix(config, translate[0], translate[1], translate[2]), config)
    # elif config.time_count[None]>33.5 and config.time_count[None]<38.5:
    #     ang = config.ang_spd[None] * (config.time_count[None] - config.time_down[None])
    #     bound.move_scene_obj('rod',rotation_matrix(config, 0, ang, 0), config)
    """ 3d_rotate_rod scene """
    if config.time_count[None] < 0.5:
        dy = -2.4 * config.dt[None]
        bound.move_scene_obj('rod',translation_matrix(config,0,dy,0), config)
    else:
        ang = math.pi * config.dt[None]
        bound.move_scene_obj('rod',rotation_matrix(config,0,0,ang), config) # rotate around z-axis

def run_step(ngrid, fluid, bound, config):
    config.time_counter[None] += 1
    actual_time_start = time.time()
    while config.time_count[None] < config.time_counter[None] / config.gui_fps[None]:
        """ computation loop """
        config.time_count[None] += config.dt[None]
        if config.solver_type == "DFSPH" or config.solver_type == "VFSPH":
            sph_step(ngrid, fluid, bound, config)
        elif config.solver_type == "JL21":
            sph_step_jl21(ngrid, fluid, bound, config)
        else:
            raise Exception('sph ERROR: no solver type', config.solver_type)
        apply_bound_transform(bound, config)
        config.frame_div_iter[None] += config.div_iter_count[None]
        config.frame_incom_iter[None] += config.incom_iter_count[None]
    config.time_consumption[None] = time.time() - actual_time_start
#################################### END SPH SOLVER ###########################################