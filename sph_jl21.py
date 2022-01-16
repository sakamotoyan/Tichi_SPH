from os import system
import numpy as np
import time

from taichi.lang.ops import atomic_min
from sph_obj import *



@ti.kernel
def JL21_cfl_condition(obj: ti.template(), config: ti.template()):
    config.dt[None] = config.part_size[1] / config.cs[None]
    # for i in range(obj.part_num[None]):
    #     v_norm = obj.vel[i].norm()
    #     if v_norm > 1e-4:
    #         atomic_min(config.dt[None], config.part_size[1] / v_norm * config.cfl_factor[None])

@ti.kernel
def JL21_clean_value(obj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    dim = ti.static(config.gravity.n)
    for i in range(obj.part_num[None]):
        obj.sph_density[i] = 0
        obj.lamb[i] = 1
        for j in ti.static(range(dim)):
            obj.acce[i][j] = 0
            obj.F_mid[i][j] = 0

@ti.kernel
def JL21_prepare_density(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
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

@ti.kernel
def JL21_pressure_val(obj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        # if abs(obj.rest_density[i]) < 1e-3:
        #     print('too small rest density')
        obj.pressure[i] = (obj.rest_density[i] * config.cs[None] ** 2 / config.wc_gamma[None]) * (
                    (obj.sph_density[i] / obj.rest_density[i]) ** config.wc_gamma[None] - 1)
        if obj.pressure[i] < 0:
            obj.pressure[i] = 0         

@ti.kernel
def JL21_pressure_force(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
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
                        # if abs(obj.sph_density[i] * nobj.sph_density[neighb_pid]) < 1e-3:
                        #     print('too small sph_density')
                        p_term = obj.mass[i] * nobj.mass[neighb_pid] / (obj.sph_density[i] * nobj.sph_density[neighb_pid]) * (obj.pressure[i] + nobj.pressure[neighb_pid])
                        if r > 1e-5:
                            obj.F_mid[i] += -p_term * xij / r * W_grad(r, config)

@ti.kernel
def JL21_update_vel_mid(obj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    dim = ti.static(config.gravity.n)
    for i in range(obj.part_num[None]):
        for j in ti.static(range(dim)):
            obj.vel_mid[i][j] = 0
        for k in ti.static(range(phase_num)):
            # if abs(obj.mass[i] * config.phase_rest_density[None][k]) < 1e-3:
            #     print('too small mass')
            obj.vel_mid_phase[i, k] = obj.vel_phase[i, k] + (obj.sph_density[i] / (obj.mass[i] * config.phase_rest_density[None][k]) 
                                    * obj.F_mid[i] + config.gravity[None]) * config.dt[None] #Eq.15
            obj.vel_mid[i] += obj.volume_frac[i][k] * obj.vel_mid_phase[i, k]

@ti.kernel
def JL21_update_vel_drag(obj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    dim = ti.static(config.gravity.n)
    for i in range(obj.part_num[None]):
        for j in ti.static(range(dim)):
            obj.vel[i][j] = 0
        for k in ti.static(range(phase_num)):
                obj.vel_phase[i, k] = obj.vel_mid_phase[i, k] - config.fbm_convection_term[None] * config.dt[None] * obj.rest_density[i] / config.phase_rest_density[None][k] * (obj.vel_mid_phase[i,k] - obj.vel_mid[i])  #Eq.16
                obj.vel[i] += obj.volume_frac[i][k] * obj.vel_phase[i, k]

@ti.kernel
def JL21_artificial_viscosity(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
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
                        if r > 1e-5:
                            # if abs(obj.sph_density[i] + nobj.sph_density[neighb_pid]) < 1e-3:
                            #     print('too small sph_density 2')
                            niu = 2 * config.artificial_viscosity[None] * config.kernel_h[1] / (obj.sph_density[i] + nobj.sph_density[neighb_pid])
                            vij = obj.vel[i] - nobj.vel[neighb_pid]
                            if vij.dot(xij) < 0:
                                obj.F_mid[i] += obj.mass[i] * nobj.mass[neighb_pid] * niu * vij.dot(xij) / (r ** 2 + 0.01 * config.kernel_h[2]) * xij / r * W_grad(r, config)


@ti.kernel
def JL21_update_pos(obj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        obj.pos[i] += obj.vel[i] * config.dt[None]

@ti.kernel
def JL21_update_color(obj: ti.template(), config: ti.template()):
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
def JL21_predict_phase_transport(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    for i in range(obj.part_num[None]):
        obj.volume_frac_tmp[i] = obj.volume_frac[i]
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
                        if r > 1e-5:
                            W_g = xij / r * W_grad(r, config) # + / - ?
                            for k in ti.static(range(phase_num)):
                                u_mk_i = obj.vel_phase[i, k] - obj.vel[i]
                                u_mk_j = nobj.vel_phase[neighb_pid, k] - nobj.vel[neighb_pid]
                                # if abs(obj.rest_density[i]) < 1e-3:
                                #     print('too small rest density 2')
                                rest_vol = obj.mass[i] / obj.rest_density[i]
                                T_m = -(obj.volume_frac[i][k] * u_mk_i + nobj.volume_frac[neighb_pid][k]* u_mk_j).dot(W_g) * rest_vol
                                T_d = 2 * config.fbm_diffusion_term[None] * (obj.volume_frac[i][k] - nobj.volume_frac[neighb_pid][k]) * rest_vol * (
                                        xij.dot(W_g) / (r ** 2 + 0.01 * config.kernel_h[2])) # + / - ?
                                obj.volume_frac_tmp[i][k] += config.dt[None] * min(obj.lamb[i], nobj.lamb[neighb_pid]) * (T_m + T_d)
        
@ti.kernel
def JL21_phase_transport_check_neg(obj: ti.template()):
    obj.general_flag[None] = 0
    for i in range(obj.part_num[None]):
        obj.flag[i] = 0
        if has_negative(obj.volume_frac_tmp[i]):
            obj.flag[i] = 1
            obj.general_flag[None] = 1

@ti.kernel
def JL21_update_lamb(obj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    for i in range(obj.part_num[None]):
        if obj.flag[i]:
            mul = 1.0
            for k in ti.static(range(phase_num)):
                if obj.volume_frac_tmp[i][k] < 0:
                    if (obj.volume_frac[i][k] - obj.volume_frac_tmp[i][k]) > 1e-5:
                        mul = min(mul, obj.volume_frac[i][k] / (obj.volume_frac[i][k] - obj.volume_frac_tmp[i][k]))
            obj.lamb[i] *= mul

@ti.kernel
def JL21_normalize_volume_frac(obj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    for i in range(obj.part_num[None]):
        s = 0.0
        for k in ti.static(range(phase_num)):
            if obj.volume_frac_tmp[i][k] < 0:
                obj.volume_frac_tmp[i][k] = 0
            s += obj.volume_frac_tmp[i][k]
        # if abs(s) < 1e-3:
        #     print('too small volume frac')
        obj.volume_frac[i] =  obj.volume_frac_tmp[i] / s            

@ti.kernel
def JL21_SPH_update_mass(obj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        obj.rest_density[i] = config.phase_rest_density[None].dot(obj.volume_frac[i])
        obj.mass[i] = obj.rest_density[i] * obj.rest_volume[i]

###################################### SPH SOLVER ############################################
def sph_step_jl21(ngrid, fluid, bound, config):
    JL21_cfl_condition(fluid, config)
    """ neighbour search """
    ngrid.clear_node(config)
    ngrid.encode(fluid, config)
    ngrid.encode(bound, config)
    ngrid.mem_shift(config)
    ngrid.fill_node(fluid, config)
    ngrid.fill_node(bound, config)
    """ SPH clean value """
    JL21_clean_value(fluid, config)
    JL21_clean_value(bound, config)
    """ WCSPH """
    JL21_prepare_density(ngrid, fluid, fluid, config)
    JL21_prepare_density(ngrid, fluid, bound, config)
    JL21_prepare_density(ngrid, bound, fluid, config)
    JL21_prepare_density(ngrid, bound, bound, config)
    JL21_pressure_val(fluid, config)
    JL21_pressure_val(bound, config)
    JL21_pressure_force(ngrid, fluid, fluid, config)
    JL21_pressure_force(ngrid, fluid, bound, config)
    JL21_artificial_viscosity(ngrid, fluid, fluid, config)
    JL21_artificial_viscosity(ngrid, fluid, bound, config)
    JL21_update_vel_mid(fluid, config)
    JL21_update_vel_drag(fluid, config)
    JL21_update_pos(fluid, config)
    for l in range(2):
        JL21_predict_phase_transport(ngrid, fluid, fluid, config)
        JL21_phase_transport_check_neg(fluid)
        if fluid.general_flag[None] == 0:
            break
        JL21_update_lamb(fluid, config)
    JL21_normalize_volume_frac(fluid, config)
    JL21_SPH_update_mass(fluid, config)
    JL21_update_color(fluid, config)

#################################### END SPH SOLVER ###########################################