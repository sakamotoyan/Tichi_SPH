# PAPER:        Divergence-free smoothed particle hydrodynamics
# AUTHOR:       Jan Bender, Dan Koschier
# CONFERENCE:   SCA '15: Proceedings of the 14th ACM SIGGRAPH / Eurographics Symposium on Computer Animation, August 2015
# URL:          https://doi.org/10.1145/2786784.2786796

import taichi as ti

from ti_sph.sim.SPH_kernel import (
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
        max_comp_iter=50,
    ):
        self.obj = obj

        self.max_div_error = ti.field(ti.f32, ())
        self.max_comp_error = ti.field(ti.f32, ())
        self.max_div_iter = ti.field(ti.i32, ())
        self.max_comp_iter = ti.field(ti.i32, ())

        self.max_div_error[None] = float(max_div_error)
        self.max_comp_error[None] = float(max_comp_error)
        self.max_div_iter[None] = int(max_div_iter)
        self.max_comp_iter[None] = int(max_comp_iter)

    def compute_density(
        self,
        obj,
        obj_pos,
        nobj,
        nobj_pos,
        nobj_mass,
        obj_output_density,
        config_neighb,
    ):
        self.compute_density_ker(
            obj,
            obj_pos,
            nobj,
            nobj_pos,
            nobj_mass,
            obj_output_density,
            config_neighb,
        )

    @ti.kernel
    def compute_density_ker(
        self,
        obj: ti.template(),
        obj_pos: ti.template(),
        nobj: ti.template(),
        nobj_pos: ti.template(),
        nobj_mass: ti.template(),
        obj_output_density: ti.template(),
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
                        obj_output_density[i] += nobj_mass[nid] * spline_W(
                            dis, obj.sph.h[i], obj.sph.sig[i]
                        )

    def compute_alpha_1(
        self,
        obj,
        obj_pos,
        nobj,
        nobj_pos,
        nobj_mass,
        obj_output_alpha_2,
        config_neighb,
    ):
        a = 1

    def compute_alpha_2(
        self,
        obj,
        obj_pos,
        nobj,
        nobj_pos,
        nobj_mass,
        obj_output_alpha_1,
        config_neighb,
    ):
        a = 1

    def compute_alpha_1_ker(
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

    @ti.kernel
    def compute_alpha_2_ker(
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
                            grad_W = (dis, obj.sph.h[i], obj.sph.sig_inv_h[i])
                            obj_output_alpha_2[i] += (
                                nobj_X[nid] * grad_W
                            ) ** 2 / nobj_mass[nid]

    @ti.kernel
    def compute_alpha_ker(
        self,
        obj: ti.template(),
        obj_mass: ti.template(),
        obj_alpha_1: ti.template(),
        obj_alpha_2: ti.template(),
        obj_output_alpha: ti.template(),
    ):
        for i in range(obj.info.stack_top[None]):
            obj_output_alpha[i] = (obj_alpha_1[i].dot(obj_alpha_1[i]) / obj_mass[i]) + obj_alpha_2[i]
            if not bigger_than_zero(obj_output_alpha[i]):
                make_bigger_than_zero(obj_output_alpha[i])
