from os import system
import numpy as np
import time

from taichi.lang.ops import atomic_min
from sph_obj import *


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
def cfl_condition(obj: ti.template(), config: ti.template()):
    config.dt[None] = config.part_size[1] / config.cs[None]
    # for i in range(obj.part_num[None]):
    #     v_norm = obj.vel[i].norm()
    #     if v_norm > 1e-4:
    #         atomic_min(config.dt[None], config.part_size[1] / v_norm * config.cfl_factor[None])

@ti.kernel
def SPH_clean_value(obj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    dim = ti.static(config.gravity.n)
    for i in range(obj.part_num[None]):
        obj.sph_density[i] = 0
        for j in ti.static(range(dim)):
            obj.acce[i][j] = 0

@ti.kernel
def SPH_prepare_density_S08(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
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
                        obj.sph_density[i] += Wr * obj.mass[i]
        # if obj.sph_density[i] < obj.rest_density[i]:
        #     obj.sph_density[i] = obj.rest_density[i]       

@ti.kernel
def WC_pressure_val(obj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        obj.pressure[i] = (obj.rest_density[i] * config.cs[None] ** 2 / config.wc_gamma[None]) * (
                    (obj.sph_density[i] / obj.rest_density[i]) ** config.wc_gamma[None] - 1)
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
                            obj.acce[i] += -p_term * nobj.mass[neighb_pid] * xij / r * W_grad(r, config)

# @ti.kernel
# def WC_pressure_acce_boundary(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
#     for i in range(obj.part_num[None]):
#         for t in range(config.neighb_search_template.shape[0]):
#             node_code = dim_encode(obj.neighb_cell_structured_seq[i] + config.neighb_search_template[t], config)
#             if 0 < node_code < config.node_num[None]:
#                 for j in range(ngrid.node_part_count[node_code]):
#                     shift = ngrid.node_part_shift[node_code] + j
#                     neighb_uid = ngrid.part_uid_in_node[shift]
#                     if neighb_uid == nobj.uid:
#                         neighb_pid = ngrid.part_pid_in_node[shift]
#                         xij = obj.pos[i] - nobj.pos[neighb_pid]
#                         r = xij.norm()
#                         p_term = 2 * obj.pressure[i] / ((obj.sph_density[i]) ** 2)
#                         if r > 0:
#                             obj.acce[i] += -p_term * nobj.mass[neighb_pid] * xij / r * W_grad(r, config)

@ti.kernel
def SPH_advection_gravity_acc(obj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        obj.acce[i] += config.gravity[None]

# @ti.kernel
# def SPH_advection_viscosity_acc(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
#     for i in range(obj.part_num[None]):
#         for t in range(config.neighb_search_template.shape[0]):
#             node_code = dim_encode(obj.neighb_cell_structured_seq[i] + config.neighb_search_template[t], config)
#             if 0 < node_code < config.node_num[None]:
#                 for j in range(ngrid.node_part_count[node_code]):
#                     shift = ngrid.node_part_shift[node_code] + j
#                     neighb_uid = ngrid.part_uid_in_node[shift]
#                     if neighb_uid == nobj.uid:
#                         neighb_pid = ngrid.part_pid_in_node[shift]
#                         xij = obj.pos[i] - nobj.pos[neighb_pid]
#                         r = xij.norm()
#                         if r > 0:
#                             vij = obj.vel[i] - nobj.vel[neighb_pid]
#                             if vij.dot(xij) < 0:
#                                 obj.acce[i] += W_lap(xij, r, nobj.mass[neighb_pid] / nobj.sph_density[neighb_pid],
#                                                      obj.vel[i] - nobj.vel[neighb_pid], config) * config.dynamic_viscosity[None] / obj.rest_density[i]

@ti.kernel
def artificial_viscosity(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
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
                            niu = 2 * config.dynamic_viscosity[None] * config.kernel_h[1] / (obj.sph_density[i] + nobj.sph_density[neighb_pid]) # using config.dynamic_viscosity[None] for tmp
                            vij = obj.vel[i] - nobj.vel[neighb_pid]
                            if vij.dot(xij) < 0:
                                obj.acce[i] += nobj.mass[neighb_pid] * niu * vij.dot(xij) / (r ** 2 + 0.01 * config.kernel_h[2]) * xij / r * W_grad(r, config)

@ti.kernel
def SPH_advection_update_vel(obj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        obj.vel[i] = obj.vel[i] + obj.acce[i] * config.dt[None]

@ti.kernel
def SPH_update_pos(obj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        obj.pos[i] += obj.vel[i] * config.dt[None]

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

###################################### SPH SOLVER ############################################
def sph_step(ngrid, fluid, bound, config):
    cfl_condition(fluid, config)
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
    """ WCSPH """
    # print(config.kernel_h[1])
    SPH_prepare_density_S08(ngrid, fluid, fluid, config)
    SPH_prepare_density_S08(ngrid, fluid, bound, config)
    SPH_prepare_density_S08(ngrid, bound, fluid, config)
    SPH_prepare_density_S08(ngrid, bound, bound, config)
    WC_pressure_val(fluid, config)
    WC_pressure_val(bound, config)
    WC_pressure_acce(ngrid, fluid, fluid, config)
    WC_pressure_acce(ngrid, fluid, bound, config)
    SPH_advection_gravity_acc(fluid, config)
    artificial_viscosity(ngrid, fluid, fluid, config)
    artificial_viscosity(ngrid, fluid, bound, config)
    SPH_advection_update_vel(fluid, config)
    # SPH_update_mass(fluid, config)
    SPH_update_pos(fluid, config)
    # SPH_update_color(fluid, config)
#################################### END SPH SOLVER ###########################################