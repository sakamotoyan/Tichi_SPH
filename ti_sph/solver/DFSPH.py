# handels DFSPH computation and error-related global variables
# PAPER:        Divergence-free smoothed particle hydrodynamics
# AUTHOR:       Jan Bender, Dan Koschier
# CONFERENCE:   SCA '15: Proceedings of the 14th ACM SIGGRAPH / Eurographics Symposium on Computer Animation, August 2015
# URL:          https://doi.org/10.1145/2786784.2786796
# uses the psi-X notation, see (Add url here)

import taichi as ti

from ti_sph.solver.SPH_kernel import (
    SPH_kernel,
    bigger_than_zero,
    make_bigger_than_zero,
    spline_W,
    spline_W_old,
    grad_spline_W,
    artificial_Laplacian_spline_W,
)


@ti.data_oriented
class DFSPH(SPH_kernel):
    def __init__(
        self,
        obj,
        max_div_error=1e-3,
        max_comp_error=1e-4,
        max_div_iter=50,
        max_comp_iter=100,
        min_comp_tier=2,
        min_div_iter=2,
    ):
        self.obj = obj

        self.max_div_error = ti.field(ti.f32, ())
        self.max_comp_error = ti.field(ti.f32, ())
        self.max_div_iter = ti.field(ti.i32, ())
        self.max_comp_iter = ti.field(ti.i32, ())
        self.min_comp_tier = ti.field(ti.i32, ())   # typo: tier -> iter
        self.min_div_iter = ti.field(ti.i32, ())
        self.comp_avg_ratio = ti.field(ti.f32, ())  # incompressible error term
        self.div_avg_ratio = ti.field(ti.f32, ())   # divergence-free error term
        self.comp_iter_count = ti.field(ti.i32, ())
        self.div_iter_count = ti.field(ti.i32, ())

        self.max_div_error[None] = float(max_div_error)
        self.max_comp_error[None] = float(max_comp_error)
        self.max_div_iter[None] = int(max_div_iter)
        self.max_comp_iter[None] = int(max_comp_iter)
        self.min_comp_tier[None] = int(min_comp_tier)
        self.min_div_iter[None] = int(min_div_iter)
        self.comp_iter_count[None] = 0
        self.div_iter_count[None] = 0
        self.comp_avg_ratio[None] = 1
        self.div_avg_ratio[None] = 1

    # returns whether incompressible solver iteration should be continued
    def is_compressible(
        self,
    ):
        return (
            (self.comp_iter_count[None] < self.min_comp_tier[None])
            or (self.comp_avg_ratio[None] > self.max_comp_error[None])
            and (not self.comp_iter_count[None] == self.max_comp_iter[None])
        )

    #compute psi (corresponds to SPH density estimationn)
    @ti.kernel
    def compute_psi(
        self,
        obj: ti.template(),
        obj_pos: ti.template(),
        nobj: ti.template(),
        nobj_pos: ti.template(),
        nobj_X: ti.template(),
        obj_output_psi: ti.template(),
        config_neighb: ti.template(),
    ):
        cell_vec = ti.static(obj.located_cell.vec)
        for i in range(obj.info.stack_top[None]):
            for cell_tpl in range(config_neighb.search_template.shape[0]):
                cell_coded = (
                    cell_vec[i] + config_neighb.search_template[cell_tpl]
                ).dot(config_neighb.cell_coder[None])
                if 0 < cell_coded < config_neighb.cell_num[None]:
                    for j in range(nobj.cell.part_count[cell_coded]):
                        shift = nobj.cell.part_shift[cell_coded] + j
                        nid = nobj.located_cell.part_log[shift]
                        """compute below"""
                        dis = (obj_pos[i] - nobj_pos[nid]).norm()
                        obj_output_psi[i] += nobj_X[nid] * spline_W(
                            dis, obj.sph.h[i], obj.sph.sig[i]
                        )

    # compute alpha_1
    @ti.kernel
    def compute_alpha_1(
        self,
        obj: ti.template(),
        obj_pos: ti.template(),
        nobj: ti.template(),
        nobj_pos: ti.template(),
        nobj_X: ti.template(),
        obj_output_alpha_1: ti.template(),
        config_neighb: ti.template(),
    ):
        cell_vec = ti.static(obj.located_cell.vec)
        for i in range(obj.info.stack_top[None]):
            for cell_tpl in range(config_neighb.search_template.shape[0]):
                cell_coded = (
                    cell_vec[i] + config_neighb.search_template[cell_tpl]
                ).dot(config_neighb.cell_coder[None])
                if 0 < cell_coded < config_neighb.cell_num[None]:
                    for j in range(nobj.cell.part_count[cell_coded]):
                        shift = nobj.cell.part_shift[cell_coded] + j
                        nid = nobj.located_cell.part_log[shift]
                        """compute below"""
                        x_ij = obj_pos[i] - nobj_pos[nid]
                        dis = x_ij.norm()
                        if dis > 1e-6:
                            grad_W_vec = (
                                grad_spline_W(dis, obj.sph.h[i], obj.sph.sig_inv_h[i])
                                * x_ij
                                / dis
                            )
                            obj_output_alpha_1[i] += nobj_X[nid] * grad_W_vec

    # compute alpha_2
    @ti.kernel
    def compute_alpha_2(
        self,
        obj: ti.template(),
        obj_pos: ti.template(),
        nobj: ti.template(),
        nobj_pos: ti.template(),
        nobj_mass: ti.template(),
        nobj_X: ti.template(),
        obj_output_alpha_2: ti.template(),
        config_neighb: ti.template(),
    ):
        cell_vec = ti.static(obj.located_cell.vec)
        for i in range(obj.info.stack_top[None]):
            for cell_tpl in range(config_neighb.search_template.shape[0]):
                cell_coded = (
                    cell_vec[i] + config_neighb.search_template[cell_tpl]
                ).dot(config_neighb.cell_coder[None])
                if 0 < cell_coded < config_neighb.cell_num[None]:
                    for j in range(nobj.cell.part_count[cell_coded]):
                        shift = nobj.cell.part_shift[cell_coded] + j
                        nid = nobj.located_cell.part_log[shift]
                        """compute below"""
                        x_ij = obj_pos[i] - nobj_pos[nid]
                        dis = x_ij.norm()
                        if dis > 1e-6:
                            grad_W = grad_spline_W(
                                dis, obj.sph.h[i], obj.sph.sig_inv_h[i]
                            )
                            obj_output_alpha_2[i] += (
                                nobj_X[nid] * grad_W
                            ) ** 2 / nobj_mass[nid]

    # compute alpha
    @ti.kernel
    def compute_alpha(
        self,
        obj: ti.template(),
        obj_mass: ti.template(),
        obj_alpha_1: ti.template(),
        obj_alpha_2: ti.template(),
        obj_output_alpha: ti.template(),
    ):
        for i in range(obj.info.stack_top[None]):
            obj_output_alpha[i] = (
                obj_alpha_1[i].dot(obj_alpha_1[i]) / obj_mass[i]
            ) + obj_alpha_2[i]
            if not bigger_than_zero(obj_output_alpha[i]):
                obj_output_alpha[i] = make_bigger_than_zero()

    # compute delta_psi
    @ti.kernel
    def compute_delta_psi(
        self,
        obj: ti.template(),
        obj_sph_psi: ti.template(),
        obj_rest_psi: ti.template(),
        obj_output_delta_psi: ti.template(),
    ):
        for i in range(obj.info.stack_top[None]):
            obj_output_delta_psi[i] = obj_sph_psi[i] - obj_rest_psi[i]

    # compute delta_psi from advection
    @ti.kernel
    def compute_adv_psi_advection(
        self,
        obj: ti.template(),
        obj_pos: ti.template(),
        obj_vel_adv: ti.template(),
        nobj: ti.template(),
        nobj_pos: ti.template(),
        nobj_vel_adv: ti.template(),
        nobj_X: ti.template(),
        dt: ti.template(),
        obj_output_delta_psi: ti.template(),
        config_neighb: ti.template(),
    ):
        cell_vec = ti.static(obj.located_cell.vec)
        for i in range(obj.info.stack_top[None]):
            for cell_tpl in range(config_neighb.search_template.shape[0]):
                cell_coded = (
                    cell_vec[i] + config_neighb.search_template[cell_tpl]
                ).dot(config_neighb.cell_coder[None])
                if 0 < cell_coded < config_neighb.cell_num[None]:
                    for j in range(nobj.cell.part_count[cell_coded]):
                        shift = nobj.cell.part_shift[cell_coded] + j
                        nid = nobj.located_cell.part_log[shift]
                        """compute below"""
                        x_ij = obj_pos[i] - nobj_pos[nid]
                        dis = x_ij.norm()
                        if dis > 1e-6:
                            obj_output_delta_psi[i] += (
                                (
                                    grad_spline_W(
                                        dis, obj.sph.h[i], obj.sph.sig_inv_h[i]
                                    )
                                    * x_ij
                                    / dis
                                ).dot(obj_vel_adv[i] - nobj_vel_adv[nid])
                                * nobj_X[nid]
                                * dt[None]
                            )

    # clamp delta_psi to above 0, AND update comp_avg_ratio (error term)
    @ti.kernel
    def statistic_non_negative_delta_psi(
        self,
        obj: ti.template(),
        obj_rest_psi: ti.template(),
        obj_output_delta_psi: ti.template(),
    ):
        for i in range(obj.info.stack_top[None]):
            if obj_output_delta_psi[i] < 0:
                obj_output_delta_psi[i] = 0

        for i in range(obj.info.stack_top[None]):
            self.comp_avg_ratio[None] += obj_output_delta_psi[i] / obj_rest_psi[i]

        self.comp_avg_ratio[None] /= obj.info.stack_top[None]

    # update vel_adv with pressure
    @ti.kernel
    def update_vel_adv(
        self,
        obj: ti.template(),
        obj_pos: ti.template(),
        obj_X: ti.template(),
        obj_delta_psi: ti.template(),
        obj_alpha: ti.template(),
        obj_mass: ti.template(),
        nobj: ti.template(),
        nobj_pos: ti.template(),
        nobj_delta_psi: ti.template(),
        nobj_X: ti.template(),
        nobj_alpha: ti.template(),
        inv_dt: ti.template(),
        obj_output_vel_adv: ti.template(),
        config_neighb: ti.template(),
    ):
        cell_vec = ti.static(obj.located_cell.vec)
        for i in range(obj.info.stack_top[None]):
            for cell_tpl in range(config_neighb.search_template.shape[0]):
                cell_coded = (
                    cell_vec[i] + config_neighb.search_template[cell_tpl]
                ).dot(config_neighb.cell_coder[None])
                if 0 < cell_coded < config_neighb.cell_num[None]:
                    for j in range(nobj.cell.part_count[cell_coded]):
                        shift = nobj.cell.part_shift[cell_coded] + j
                        nid = nobj.located_cell.part_log[shift]
                        """compute below"""
                        x_ij = obj_pos[i] - nobj_pos[nid]
                        dis = x_ij.norm()
                        if dis > 1e-6:
                            obj_output_vel_adv[i] += (
                                -inv_dt[None]
                                * (
                                    grad_spline_W(
                                        dis, obj.sph.h[i], obj.sph.sig_inv_h[i]
                                    )
                                    * x_ij
                                    / dis
                                )
                                / obj_mass[i]
                                * (
                                    (obj_delta_psi[i] * nobj_X[nid] / obj_alpha[i])
                                    + (nobj_delta_psi[nid] * obj_X[i] / nobj_alpha[nid])
                                )
                            )
