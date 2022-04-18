# PAPER:    An Implicit SPH Formulation for Incompressible Linearly Elastic Solids
# AUTHOR:   Andreas Peer, Christoph Gissler, Stefan Band, Matthias Teschner
# JOURNAL:  COMPUTER GRAPHICS FORUM Volume37, Issue6
# URL:      https://onlinelibrary.wiley.com/doi/10.1111/cgf.13317

import taichi as ti
from .SPH_kernel import *
import numpy as np


@ti.data_oriented
class ISPH_Elastic:
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
    def compute_F(
        self,
        obj_volume,
        obj_pos_0,
        obj_pos_now,
        obj_L,
        obj_output_F,
        config_neighb,
    ):
        self.compute_F_ker(
            self.obj,
            obj_volume,
            obj_pos_0,
            obj_pos_now,
            obj_L,
            obj_output_F,
            config_neighb,
        )

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

    # Eqn.1
    def compute_L(
        self,
        obj_volume,
        obj_pos_0,
        obj_output_L,
        config_neighb,
    ):
        self.compute_L_ker(
            self.obj,
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

    def compute_R_pd(
        self,
        obj_F,
        obj_output_R,
    ):
        self.compute_R_pd_ker(self.obj, obj_F, obj_output_R)

    @ti.kernel
    def compute_R_pd_ker(
        self,
        obj: ti.template(),
        obj_F: ti.template(),
        obj_output_R: ti.template(),
    ):
        for i in range(obj.info.stack_top[None]):
            obj_output_R[i] = ti.polar_decompose(obj_F[i])[0]

    # Eqn.5
    def compute_F_star(
        self,
        obj_volume,
        obj_pos_0,
        obj_pos_now,
        obj_R,
        obj_L,
        obj_output_F_star,
        config_neighb,
    ):
        self.compute_F_star_ker(
            self.obj,
            obj_volume,
            obj_pos_0,
            obj_pos_now,
            obj_R,
            obj_L,
            obj_output_F_star,
            config_neighb,
        )

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

    def compute_eps(
        self,
        obj_F,
        obj_output_eps,
    ):
        self.compute_eps_ker(
            self.obj,
            obj_F,
            obj_output_eps,
        )

    @ti.kernel
    def compute_eps_ker(
        self,
        obj: ti.template(),
        obj_F: ti.template(),
        obj_output_eps: ti.template(),
    ):
        I = ti.Matrix.identity(dt=ti.f32, n=obj_F[0].n)
        for i in range(obj.info.stack_top[None]):
            obj_output_eps[i] = (obj_F[i] + obj_F[i].transpose()) * 0.5 - I

    def compute_P(
        self,
        obj_eps,
        obj_output_P,
    ):
        self.compute_P_ker(
            self.obj,
            obj_eps,
            obj_output_P,
        )

    @ti.kernel
    def compute_P_ker(
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

    def compute_force(
        self,
        obj_volume,
        obj_pos_0,
        obj_R,
        obj_L,
        obj_P,
        obj_output_force,
        config_neighb,
    ):
        self.compute_force_ker(
            self.obj,
            obj_volume,
            obj_pos_0,
            obj_R,
            obj_L,
            obj_P,
            obj_output_force,
            config_neighb,
        )

    @ti.kernel
    def compute_force_ker(
        self,
        obj: ti.template(),
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
                        if not nid == i:
                            # compute down below
                            x_ji_0 = obj_pos_0[nid] - obj_pos_0[i]
                            dis_0 = distance_2(x_ji_0)
                            grad_W_vec_i = (
                                obj_R[i]
                                @ obj_L[i]
                                @ (
                                    grad_spline_W(dis_0, obj.sph.h[i], obj.sph.sig[i])
                                    * (-x_ji_0)
                                    / dis_0
                                )
                            )
                            grad_W_vec_nid = (
                                obj_R[nid]
                                @ obj_L[nid]
                                @ (
                                    grad_spline_W(
                                        dis_0, obj.sph.h[nid], obj.sph.sig[nid]
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

    # def update_acc(
    #     self,
    #     obj_force,
    #     obj_mass,
    #     obj_output_acc,
    # ):
    #     self.update_acc_ker(
    #         self.obj,
    #         obj_force,
    #         obj_mass,
    #         obj_output_acc,
    #     )

    # @ti.kernel
    # def update_acc_ker(
    #     self,
    #     obj: ti.template(),
    #     obj_force: ti.template(),
    #     obj_mass: ti.template(),
    #     obj_output_acc: ti.template(),
    # ):
    #     for i in range(obj.info.stack_top[None]):
    #         obj_output_acc[i] += obj_force[i] / obj_mass[i]

    # def compute_vel(
    #     self,
    #     dt,
    #     obj_acc,
    #     obj_output_vel,
    # ):
    #     self.compute_vel_ker(
    #         self.obj,
    #         dt,
    #         obj_acc,
    #         obj_output_vel,
    #     )

    # @ti.kernel
    # def compute_vel_ker(
    #     self,
    #     obj: ti.template(),
    #     dt: ti.f32,
    #     obj_acc: ti.template(),
    #     obj_output_vel: ti.template(),
    # ):
    #     for i in range(obj.info.stack_top[None]):
    #         obj_output_vel[i] += obj_acc[i] * dt
