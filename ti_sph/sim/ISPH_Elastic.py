# PAPER:    An Implicit SPH Formulation for Incompressible Linearly Elastic Solids
# AUTHOR:   Andreas Peer, Christoph Gissler, Stefan Band, Matthias Teschner
# JOURNAL:  COMPUTER GRAPHICS FORUM Volume37, Issue6
# URL:      https://onlinelibrary.wiley.com/doi/10.1111/cgf.13317

import taichi as ti
from .SPH_kernel import *
import numpy as np


@ti.data_oriented
class ISPH_Elastic:
    def __init__(self):
        pass

    # Eqn.3
    def compute_F(
        self,
        obj,
        obj_volume,
        obj_pos_0,
        obj_pos_now,
        obj_L,
        obj_output_F,
        config_neighb,
    ):
        self.compute_F_ker(
            obj,
            obj_volume,
            obj_pos_0,
            obj_pos_now,
            obj_L,
            obj_output_F,
            config_neighb,
        )

    # Eqn.5
    def compute_F_star(
        self,
        obj,
        obj_volume,
        obj_pos_0,
        obj_pos_now,
        obj_R,
        obj_L,
        obj_output_F_star,
        config_neighb,
    ):
        self.compute_F_star_ker(
            obj,
            obj_volume,
            obj_pos_0,
            obj_pos_now,
            obj_R,
            obj_L,
            obj_output_F_star,
            config_neighb,
        )

    # Eqn.1
    def compute_L(
        self,
        obj,
        obj_volume,
        obj_pos_0,
        obj_output_L,
        config_neighb,
    ):
        self.compute_L_ker(
            obj,
            obj_volume,
            obj_pos_0,
            obj_output_L,
            config_neighb,
        )
        L = obj_output_L.to_numpy()
        inv = np.linalg.pinv(L[: obj.info.stack_top[None]])
        L[: obj.info.stack_top[None]] = inv
        obj_output_L.from_numpy(L)

    def compute_R_pd(
        self,
        obj,
        obj_F,
        obj_output_R,
    ):
        self.compute_R_pd_ker(obj, obj_F, obj_output_R)

    @ti.kernel
    def compute_F_ker(
        self,
        obj: ti.template(),
        obj_volume: ti.template(),
        obj_pos_0: ti.template(),
        obj_pos_now: ti.template(),
        obj_L: ti.template(),
        obj_output_F: ti.template(),
        config_neighb: ti.template(),
    ):
        cell_vec = ti.static(obj.located_cell.vec)
        for i in range(obj.info.stack_top[None]):
            for cell_tpl in range(config_neighb.search_template.shape[0]):
                cell_coded = (
                    cell_vec[i] + config_neighb.search_template[cell_tpl]
                ).dot(config_neighb.cell_coder[None])
                if 0 < cell_coded < config_neighb.cell_num[None]:
                    for j in range(obj.cell.part_count[cell_coded]):
                        shift = obj.cell.part_shift[cell_coded] + j
                        nid = obj.located_cell.part_log[shift]
                        if not nid == i:
                            # compute down below
                            x_ji_0 = obj_pos_0[nid] - obj_pos_0[i]
                            x_ji_now = obj_pos_now[nid] - obj_pos_now[i]
                            dis_0 = distance_2(x_ji_0)
                            grad_W_vec = obj_L[i] @ (
                                grad_spline_W(dis_0, obj.sph.h[i], obj.sph.sig[i])
                                * (-x_ji_0)
                                / dis_0
                            )
                            obj_output_F[i] += obj_volume[nid] * x_ji_now.outer_product(
                                grad_W_vec
                            )

    @ti.kernel
    def compute_L_ker(
        self,
        obj: ti.template(),
        obj_volume: ti.template(),
        obj_pos_0: ti.template(),
        obj_output_L: ti.template(),
        config_neighb: ti.template(),
    ):
        cell_vec = ti.static(obj.located_cell.vec)
        for i in range(obj.info.stack_top[None]):
            for cell_tpl in range(config_neighb.search_template.shape[0]):
                cell_coded = (
                    cell_vec[i] + config_neighb.search_template[cell_tpl]
                ).dot(config_neighb.cell_coder[None])
                if 0 < cell_coded < config_neighb.cell_num[None]:
                    for j in range(obj.cell.part_count[cell_coded]):
                        shift = obj.cell.part_shift[cell_coded] + j
                        nid = obj.located_cell.part_log[shift]
                        if not nid == i:
                            # compute down below
                            x_ji_0 = obj_pos_0[nid] - obj_pos_0[i]
                            dis_0 = distance_2(x_ji_0)
                            grad_W_vec = (
                                grad_spline_W(dis_0, obj.sph.h[i], obj.sph.sig[i])
                                * (-x_ji_0)
                                / dis_0
                            )
                            obj_output_L[i] += obj_volume[
                                nid
                            ] * grad_W_vec.outer_product(x_ji_0)

    @ti.kernel
    def compute_R_pd_ker(
        self,
        obj: ti.template(),
        obj_F: ti.template(),
        obj_output_R: ti.template(),
    ):
        for i in range(obj.info.stack_top[None]):
            obj_output_R[i] = ti.polar_decompose(obj_F[i])[0]

    @ti.kernel
    def compute_F_star_ker(
        self,
        obj: ti.template(),
        obj_volume: ti.template(),
        obj_pos_0: ti.template(),
        obj_pos_now: ti.template(),
        obj_R: ti.template(),
        obj_L: ti.template(),
        obj_output_F_star: ti.template(),
        config_neighb: ti.template(),
    ):
        dim = ti.static(obj.basic.pos[0].n)
        I = ti.Matrix.identity(dt=ti.f32, n=dim)
        cell_vec = ti.static(obj.located_cell.vec)
        for i in range(obj.info.stack_top[None]):
            for cell_tpl in range(config_neighb.search_template.shape[0]):
                cell_coded = (
                    cell_vec[i] + config_neighb.search_template[cell_tpl]
                ).dot(config_neighb.cell_coder[None])
                if 0 < cell_coded < config_neighb.cell_num[None]:
                    for j in range(obj.cell.part_count[cell_coded]):
                        shift = obj.cell.part_shift[cell_coded] + j
                        nid = obj.located_cell.part_log[shift]
                        if not nid == i:
                            # compute down below
                            x_ji_0 = obj_pos_0[nid] - obj_pos_0[i]
                            x_ji_now = obj_pos_now[nid] - obj_pos_now[i]
                            dis_0 = distance_2(x_ji_0)
                            grad_W_vec = (
                                obj_R[i]
                                @ obj_L[i]
                                @ (
                                    grad_spline_W(dis_0, obj.sph.h[i], obj.sph.sig[i])
                                    * (-x_ji_0)
                                    / dis_0
                                )
                            )
                            obj_output_F_star[i] += obj_volume[nid] * (
                                x_ji_now - (obj_R[i] @ x_ji_0)
                            ).outer_product(grad_W_vec)
            obj_output_F_star[i] += I
