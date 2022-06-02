# PAPER:    An Implicit SPH Formulation for Incompressible Linearly Elastic Solids
# AUTHOR:   Andreas Peer, Christoph Gissler, Stefan Band, Matthias Teschner
# JOURNAL:  COMPUTER GRAPHICS FORUM Volume37, Issue6
# URL:      https://onlinelibrary.wiley.com/doi/10.1111/cgf.13317

import taichi as ti
from .SPH_kernel import *
import numpy as np


@ti.data_oriented
class ISPH_Elastic(SPH_kernel):
    def __init__(self, obj, K=2e5, G=2e5):
        if not "node_ISPH_Elastic" in obj.capacity_list:
            print("Exception for the construction of ISPH_Elastic():")
            print("obj does not have the capacity 'node_ISPH_Elastic'")
            exit(0)

        self.obj = obj
        self.K = ti.field(ti.f32, ())
        self.G = ti.field(ti.f32, ())
        self.K[None] = float(K)
        self.G[None] = float(G)
        pass

    # Eqn.3
    @ti.kernel
    def compute_F(
        self,
        obj: ti.template(),
        obj_sph: ti.template(),
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
                        # compute down below
                        x_ji_0 = obj_pos_0[nid] - obj_pos_0[i]
                        dis_0 = x_ji_0.norm()
                        if dis_0 > 1e-6:
                            x_ji_now = obj_pos_now[nid] - obj_pos_now[i]
                            grad_W_vec = obj_L[i] @ (
                                grad_spline_W(dis_0, obj_sph.h[i], obj_sph.sig_inv_h[i])
                                * (-x_ji_0)
                                / dis_0
                            )
                            obj_output_F[i] += obj_volume[nid] * x_ji_now.outer_product(
                                grad_W_vec
                            )

    # Eqn.1
    def compute_L(
        self,
        obj_sph,
        obj_volume,
        obj_pos_0,
        obj_output_L,
        config_neighb,
    ):
        self.compute_L_ker(
            self.obj,
            obj_sph,
            obj_volume,
            obj_pos_0,
            obj_output_L,
            config_neighb,
        )
        L = obj_output_L.to_numpy()
        inv = np.linalg.pinv(L[: self.obj.info.stack_top[None]])
        L[: self.obj.info.stack_top[None]] = inv
        obj_output_L.from_numpy(L)

    @ti.kernel
    def compute_L_ker(
        self,
        obj: ti.template(),
        obj_sph: ti.template(),
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
                        # compute down below
                        x_ji_0 = obj_pos_0[nid] - obj_pos_0[i]
                        dis_0 = x_ji_0.norm()
                        if dis_0 > 1e-6:
                            grad_W_vec = (
                                grad_spline_W(dis_0, obj_sph.h[i], obj_sph.sig_inv_h[i])
                                * (-x_ji_0)
                                / dis_0
                            )
                            obj_output_L[i] += obj_volume[
                                nid
                            ] * grad_W_vec.outer_product(x_ji_0)

    @ti.kernel
    def compute_R_pd(
        self,
        obj: ti.template(),
        obj_F: ti.template(),
        obj_output_R: ti.template(),
    ):
        for i in range(obj.info.stack_top[None]):
            obj_output_R[i] = ti.polar_decompose(obj_F[i])[0]

    # Eqn.5
    @ti.kernel
    def compute_F_star(
        self,
        obj: ti.template(),
        obj_sph: ti.template(),
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
                        # compute down below
                        x_ji_0 = obj_pos_0[nid] - obj_pos_0[i]
                        dis_0 = x_ji_0.norm()
                        if dis_0 > 1e-6:
                            x_ji_now = obj_pos_now[nid] - obj_pos_now[i]
                            grad_W_vec = (
                                obj_R[i]
                                @ obj_L[i]
                                @ (
                                    grad_spline_W(
                                        dis_0, obj_sph.h[i], obj_sph.sig_inv_h[i]
                                    )
                                    * (-x_ji_0)
                                    / dis_0
                                )
                            )
                            obj_output_F_star[i] += obj_volume[nid] * (
                                x_ji_now - (obj_R[i] @ x_ji_0)
                            ).outer_product(grad_W_vec)
            obj_output_F_star[i] += I

    @ti.kernel
    def compute_eps(
        self,
        obj: ti.template(),
        obj_F: ti.template(),
        obj_output_eps: ti.template(),
    ):
        I = ti.Matrix.identity(dt=ti.f32, n=obj_F[0].n)
        for i in range(obj.info.stack_top[None]):
            obj_output_eps[i] = (obj_F[i] + obj_F[i].transpose()) * 0.5 - I

    @ti.kernel
    def compute_P(
        self,
        obj: ti.template(),
        obj_eps: ti.template(),
        obj_output_P: ti.template(),
    ):
        I = ti.Matrix.identity(dt=ti.f32, n=obj_eps[0].n)
        for i in range(obj.info.stack_top[None]):
            obj_output_P[i] = (2 * self.G[None] * obj_eps[i]) + (
                (self.K[None] - (2 / 3 * self.G[None])) * obj_eps[i].trace() * I
            )

    @ti.kernel
    def compute_force(
        self,
        obj: ti.template(),
        obj_sph: ti.template(),
        obj_volume: ti.template(),
        obj_pos_0: ti.template(),
        obj_R: ti.template(),
        obj_L: ti.template(),
        obj_P: ti.template(),
        obj_output_force: ti.template(),
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
                        # compute down below
                        x_ji_0 = obj_pos_0[nid] - obj_pos_0[i]
                        dis_0 = x_ji_0.norm()
                        if dis_0 > 1e-6:
                            grad_W_vec_i = (
                                obj_R[i]
                                @ obj_L[i]
                                @ (
                                    grad_spline_W(
                                        dis_0, obj_sph.h[i], obj_sph.sig_inv_h[i]
                                    )
                                    * (-x_ji_0)
                                    / dis_0
                                )
                            )
                            grad_W_vec_nid = (
                                obj_R[nid]
                                @ obj_L[nid]
                                @ (
                                    grad_spline_W(
                                        dis_0, obj_sph.h[nid], obj_sph.sig_inv_h[nid]
                                    )
                                    * (x_ji_0)
                                    / dis_0
                                )
                            )
                            obj_output_force[i] += (
                                obj_volume[i]
                                * obj_volume[nid]
                                * (
                                    (obj_P[i] @ grad_W_vec_i)
                                    - (obj_P[nid] @ grad_W_vec_nid)
                                )
                            )
