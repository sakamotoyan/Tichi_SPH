from sph_obj import *


@ti.kernel
def FBM_clean_tmp(obj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    for i in range(obj.part_num[None]):
        for j in ti.static(range(phase_num)):
            obj.volume_frac_tmp[i][j] = 0


@ti.kernel
def FBM_clean_value(obj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    dim = ti.static(config.gravity.n)
    for i in range(obj.part_num[None]):
        for k in ti.static(range(phase_num)):
            obj.phase_acc[i, k] = ti.Vector([0, 0, 0])


@ti.kernel
def FBM_correct_vel_from_phase_vel(obj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    dim = ti.static(config.gravity.n)
    for i in range(obj.part_num[None]):
        for k in ti.static(range(phase_num)):
            if obj.volume_frac[i][k] < 1e-6:
                obj.phase_vel[i, k] = obj.vel_adv[i]
    for i in range(obj.part_num[None]):
        obj.vel_adv[i] *= 0
        for k in ti.static(range(phase_num)):
            obj.vel_adv[i] += obj.volume_frac[i][k] * obj.phase_vel[i, k]
        for k in ti.static(range(phase_num)):
            obj.drift_vel[i, k] = obj.phase_vel[i, k] - obj.vel_adv[i]


@ti.kernel
def FBM_advection_M_vis(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    for i in range(obj.part_num[None]):
        for t in range(config.neighb_search_template.shape[0]):
            node_code = dim_encode(
                obj.neighb_cell_structured_seq[i] + config.neighb_search_template[t], config)
            if 0 < node_code < config.node_num[None]:
                for j in range(ngrid.node_part_count[node_code]):
                    shift = ngrid.node_part_shift[node_code] + j
                    neighb_uid = ngrid.part_uid_in_node[shift]
                    if neighb_uid == nobj.uid:
                        neighb_pid = ngrid.part_pid_in_node[shift]
                        xij = obj.pos[i] - nobj.pos[neighb_pid]
                        r = xij.norm()
                        if r > 0:
                            for k in ti.static(range(phase_num)):
                                obj.phase_acc[i, k] += config.fbm_convection_term[None] * W_lap(xij, r, nobj.X[neighb_pid] / nobj.sph_psi[neighb_pid], (obj.phase_vel[i, k] - 
                                    nobj.vel[neighb_pid]), config) * config.dynamic_viscosity[None] / config.phase_rest_density[None][k]
                                obj.phase_acc[i, k] += (1 - config.fbm_convection_term[None]) * W_lap(xij, r, nobj.X[neighb_pid] / nobj.sph_psi[neighb_pid],
                                                     obj.vel[i] - nobj.vel[neighb_pid], config) * config.dynamic_viscosity[None] / obj.rest_density[i]


@ti.kernel
def FBM_acc_2_phase_vel(obj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    dim = ti.static(config.gravity.n)
    for i in range(obj.part_num[None]):
        for k in ti.static(range(phase_num)):
            obj.phase_vel[i, k] += obj.phase_acc[i, k] * config.dt[None]


@ti.kernel
def FBM_convect(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    for i in range(obj.part_num[None]):
        obj.acce_adv[i] = (obj.vel_adv[i] - obj.vel[i]) / config.dt[None]
        obj.fbm_zeta[i] = 0
        for j in ti.static(range(phase_num)):
            obj.fbm_zeta[i] += obj.volume_frac[i][j] * (
                config.phase_rest_density[None][j] - obj.rest_density[i]) / config.phase_rest_density[None][j]
        obj.fbm_acce[i] = (obj.acce_adv[i] - (obj.fbm_zeta[i]
                           * config.gravity[None])) / (1 - obj.fbm_zeta[i])
        for j in ti.static(range(phase_num)):
            drift_vel_tmp = obj.drift_vel[i, j]
            obj.drift_vel[i, j] = (config.phase_rest_density[None][j] - obj.rest_density[i]) * (
                config.gravity[None] - obj.fbm_acce[i]) / config.phase_rest_density[None][j]
            obj.drift_vel[i, j] *= config.dt[None]
            obj.drift_vel[i, j] += drift_vel_tmp
            if obj.volume_frac[i][j] < 1e-6:
                obj.drift_vel[i, j] *= 0
            obj.phase_vel[i, j] = obj.vel_adv[i] + obj.drift_vel[i, j]


@ti.kernel
def FBM_change_tmp(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    for i in range(obj.part_num[None]):
        if obj.flag[i] == 0:  # flag check
            for t in range(config.neighb_search_template.shape[0]):
                node_code = dim_encode(
                    obj.neighb_cell_structured_seq[i] + config.neighb_search_template[t], config)
                if 0 < node_code < config.node_num[None]:
                    for j in range(ngrid.node_part_count[node_code]):
                        shift = ngrid.node_part_shift[node_code] + j
                        neighb_uid = ngrid.part_uid_in_node[shift]
                        if neighb_uid == nobj.uid:
                            neighb_pid = ngrid.part_pid_in_node[shift]
                            if nobj.flag[neighb_pid] == 0:  # flag check
                                xij = obj.pos[i] - nobj.pos[neighb_pid]
                                r = xij.norm()
                                if r > 0:
                                    for k in ti.static(range(phase_num)):
                                        tmp = config.fbm_convection_term[None] * config.dt[None] * nobj.rest_volume[neighb_pid] * (
                                            obj.volume_frac[i][k] * obj.drift_vel[i, k] + nobj.volume_frac[neighb_pid][k] * nobj.drift_vel[neighb_pid, k]).dot(xij / r) * W_grad(r, config)
                                        obj.volume_frac_tmp[i][k] -= tmp


@ti.kernel
def FBM_check_tmp(obj: ti.template()):
    obj.general_flag[None] = 0
    for i in range(obj.part_num[None]):
        if has_negative(obj.volume_frac[i] + obj.volume_frac_tmp[i]):
            obj.flag[i] = 1
            obj.general_flag[None] = 1


@ti.kernel
def FBM_update_volume_frac(obj: ti.template()):
    for i in range(obj.part_num[None]):
        if not obj.flag[i] > 0:
            obj.volume_frac[i] += obj.volume_frac_tmp[i]


@ti.kernel
def FBM_momentum_exchange_1(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    for i in range(obj.part_num[None]):
        if obj.flag[i] == 0:  # flag check
            for k in ti.static(range(phase_num)):
                obj.phase_acc[i, k] = obj.volume_frac[i][k] * \
                    obj.phase_vel[i, k]
            for t in range(config.neighb_search_template.shape[0]):
                node_code = dim_encode(
                    obj.neighb_cell_structured_seq[i] + config.neighb_search_template[t], config)
                if 0 < node_code < config.node_num[None]:
                    for j in range(ngrid.node_part_count[node_code]):
                        shift = ngrid.node_part_shift[node_code] + j
                        neighb_uid = ngrid.part_uid_in_node[shift]
                        if neighb_uid == nobj.uid:
                            neighb_pid = ngrid.part_pid_in_node[shift]
                            if nobj.flag[neighb_pid] == 0:  # flag check
                                xij = obj.pos[i] - nobj.pos[neighb_pid]
                                r = xij.norm()
                                if r > 0:
                                    for k in ti.static(range(phase_num)):
                                        tmp = config.fbm_convection_term[None] * config.dt[None] * nobj.rest_volume[neighb_pid] * (
                                            obj.volume_frac[i][k] * obj.drift_vel[i, k] + nobj.volume_frac[neighb_pid][k] * nobj.drift_vel[neighb_pid, k]).dot(xij / r) * W_grad(r, config)
                                        if tmp < 0:
                                            obj.phase_acc[i, k] -= tmp * \
                                                nobj.phase_vel[neighb_pid, k]
                                        else:
                                            obj.phase_acc[i, k] -= tmp * \
                                                obj.phase_vel[neighb_pid, k]


@ti.kernel
def FBM_momentum_exchange_2(obj: ti.template(), config: ti.template()):
    phase_num = ti.static(config.phase_rest_density.n)
    for i in range(obj.part_num[None]):
        if obj.flag[i] == 0:
            for k in ti.static(range(phase_num)):
                if obj.volume_frac[i][k] < 1e-6:
                    obj.phase_vel[i, k] = obj.vel_adv[i]
                else:
                    obj.phase_vel[i, k] = obj.phase_acc[i, k] / \
                        obj.volume_frac[i][k]


@ti.kernel
def FBM_diffuse(ngrid: ti.template(), obj: ti.template(), nobj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        if obj.flag[i] == 0:  # flag check
            for t in range(config.neighb_search_template.shape[0]):
                node_code = dim_encode(
                    obj.neighb_cell_structured_seq[i] + config.neighb_search_template[t], config)
                if 0 < node_code < config.node_num[None]:
                    for j in range(ngrid.node_part_count[node_code]):
                        shift = ngrid.node_part_shift[node_code] + j
                        neighb_uid = ngrid.part_uid_in_node[shift]
                        if neighb_uid == nobj.uid:
                            neighb_pid = ngrid.part_pid_in_node[shift]
                            if nobj.flag[neighb_pid] == 0:  # flag check
                                xij = obj.pos[i] - nobj.pos[neighb_pid]
                                r = xij.norm()
                                if r > 0:
                                    tmp = config.dt[None] * config.fbm_diffusion_term[None] * (
                                        obj.volume_frac[i] - nobj.volume_frac[neighb_pid]) * nobj.rest_volume[neighb_pid] * r * W_grad(r, config) / (r ** 2 + 0.01 * config.kernel_h[2])
                                    obj.volume_frac_tmp[i] += tmp


@ti.kernel
def debug_volume_frac(obj: ti.template()) -> ti.f32:
    p1 = 0.0
    for i in range(obj.part_num[None]):
        p1 += obj.volume_frac[i][0]
    return p1/obj.part_num[None]
