from os import system
import numpy as np
import time

from taichi.lang.ops import atomic_min
from sph_obj import *


@ti.kernel
def SPH_neighbour_loop_template(ngrid: ti.template(), obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(neighb_template.shape[0]):
            node_code = dim_encode(obj.node[i]+neighb_template[t])
            if 0 < node_code < node_num:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code]+j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        obj.sph_psi[i] += nobj.rest_volume[neighb_pid]*W((obj.pos[i] - nobj.pos[neighb_pid]).norm())

@ti.kernel
def SPH_clean_value(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.W[i] = 0
        obj.sph_compression[i] = 0
        obj.sph_density[i] = 0
        obj.alpha_2[i] = 0
        for j in ti.static(range(dim)):
            obj.W_grad[i][j] = 0
            obj.acce[i][j] = 0
            obj.acce_adv[i][j] = 0
            obj.alpha_1[i][j] = 0
            obj.pressure_force[i][j] = 0

@ti.kernel
def cfl_condition(obj: ti.template()):
    dt[None] = init_part_size/cs
    for i in range(obj.part_num[None]):
        v_norm = obj.vel[i].norm()
        if v_norm > 1e-4:
            atomic_min(dt[None], part_size[1]/v_norm*cfl_factor)

@ti.kernel
def SPH_prepare_attr(ngrid: ti.template(), obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(neighb_template.shape[0]):
            node_code = dim_encode(obj.node[i]+neighb_template[t])
            if 0 < node_code < node_num:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code]+j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        Wr = W((obj.pos[i] - nobj.pos[neighb_pid]).norm())
                        obj.W[i] += Wr
                        obj.sph_compression[i] += Wr*nobj.rest_volume[neighb_pid]
                        obj.sph_density[i] += Wr*nobj.mass[neighb_pid]

@ti.kernel
def SPH_prepare_alpha_1(ngrid: ti.template(), obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(neighb_template.shape[0]):
            node_code = dim_encode(obj.node[i]+neighb_template[t])
            if 0 < node_code < node_num:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code]+j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        xij = obj.pos[i] - nobj.pos[neighb_pid]
                        r = xij.norm()
                        if r>0:
                            obj.alpha_1[i] += nobj.X[neighb_pid] * xij/r*W_grad(r)

@ti.kernel
def SPH_prepare_alpha_2(ngrid: ti.template(), obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(neighb_template.shape[0]):
            node_code = dim_encode(obj.node[i]+neighb_template[t])
            if 0 < node_code < node_num:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code]+j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        r = (obj.pos[i] - nobj.pos[neighb_pid]).norm()
                        if r>0:
                            obj.alpha_2[i] += W_grad(r)**2 * nobj.X[neighb_pid]**2 / nobj.mass[neighb_pid]

@ti.kernel
def SPH_prepare_alpha(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.alpha[i] = obj.alpha_1[i].dot(obj.alpha_1[i])/obj.mass[i] + obj.alpha_2[i]
        if obj.alpha[i]<1e-4:
            obj.alpha[i]=1e-4

@ti.kernel
def SPH_advection_gravity_acc(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.acce_adv[i] += gravity[None]

@ti.kernel
def SPH_advection_viscosity_acc(ngrid: ti.template(), obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(neighb_template.shape[0]):
            node_code = dim_encode(obj.node[i]+neighb_template[t])
            if 0 < node_code < node_num:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code]+j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        xij = obj.pos[i] - nobj.pos[neighb_pid]
                        r = xij.norm()
                        if r>0:
                            obj.acce_adv[i] += W_lap(xij, r, nobj.X[neighb_pid]/nobj.sph_psi[neighb_pid], obj.vel[i] - nobj.vel[neighb_pid])*dynamic_viscosity/obj.rest_density[i]

@ti.kernel
def WC_pressure_val(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.pressure[i] = (obj.rest_density[i] * cs**2 / wc_gamma)*((obj.sph_density[i]/obj.rest_density[i])**7-1)
        if obj.pressure[i]<0:
            obj.pressure[i]=0
@ti.kernel
def WC_pressure_acce(ngrid: ti.template(), obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(neighb_template.shape[0]):
            node_code = dim_encode(obj.node[i]+neighb_template[t])
            if 0 < node_code < node_num:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code]+j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        xij = obj.pos[i] - nobj.pos[neighb_pid]
                        r = xij.norm()
                        p_term = obj.pressure[i]/((obj.sph_density[i])**2) + nobj.pressure[neighb_pid]/((nobj.sph_density[neighb_pid])**2)
                        if r>0:
                            obj.acce_adv[i] += -p_term * nobj.mass[neighb_pid] * xij/r * W_grad(r)

@ti.kernel
def IPPE_adv_psi_init(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.psi_adv[i] = obj.sph_psi[i] - obj.rest_psi[i]

@ti.kernel
def IPPE_adv_psi(ngrid: ti.template(), obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(neighb_template.shape[0]):
            node_code = dim_encode(obj.node[i]+neighb_template[t])
            if 0 < node_code < node_num:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code]+j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        xij = obj.pos[i] - nobj.pos[neighb_pid]
                        r = xij.norm()
                        if r>0:
                            obj.psi_adv[i] += (xij/r*W_grad(r)).dot(obj.vel_adv[i] - nobj.vel_adv[neighb_pid]) * nobj.X[neighb_pid] * dt[None]

@ti.kernel
def IPPE_psi_adv_non_negative(obj: ti.template()):
    obj.compression[None] = 0
    for i in range(obj.part_num[None]):
        if obj.psi_adv[i]<0:
            obj.psi_adv[i] = 0
        obj.compression[None] += (obj.psi_adv[i] / obj.rest_psi[i])
    obj.compression[None] /= obj.part_num[None]
    
@ti.kernel
def IPPE_update_vel_adv(ngrid: ti.template(), obj: ti.template(), nobj: ti.template()):
    for i in range(obj.part_num[None]):
        for t in range(neighb_template.shape[0]):
            node_code = dim_encode(obj.node[i]+neighb_template[t])
            if 0 < node_code < node_num:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code]+j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        xij = obj.pos[i] - nobj.pos[neighb_pid]
                        r = xij.norm()
                        if r>0:
                            obj.vel_adv[i] += -(1/dt[None]) * ((obj.psi_adv[i]*nobj.X[neighb_pid]/obj.alpha[i])+(
                                nobj.psi_adv[neighb_pid]*obj.X[i]/nobj.alpha[neighb_pid])) * (xij/r*W_grad(r)) / obj.mass[i]

@ti.kernel
def SPH_advection_update_vel_adv(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.vel_adv[i] = obj.vel[i] + obj.acce_adv[i]*dt[None]

@ti.kernel
def SPH_vel_2_vel_adv(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.vel_adv[i] = obj.vel[i]

@ti.kernel
def SPH_vel_adv_2_vel(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.vel[i] = obj.vel_adv[i]

@ti.kernel
def SPH_update_pos(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.vel[i] = obj.vel_adv[i]
        obj.pos[i] += obj.vel[i]*dt[None]
