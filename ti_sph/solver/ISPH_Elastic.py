# PAPER:    An Implicit SPH Formulation for Incompressible Linearly Elastic Solids
# AUTHOR:   Andreas Peer, Christoph Gissler, Stefan Band, Matthias Teschner
# JOURNAL:  COMPUTER GRAPHICS FORUM Volume37, Issue6
# URL:      https://onlinelibrary.wiley.com/doi/10.1111/cgf.13317

from unittest import result
import taichi as ti
from .SPH_kernel import *
import numpy as np


@ti.data_oriented
class ISPH_Elastic(SPH_kernel):
    def __init__(
        self,
        obj,
        dt,
        background_neighb_grid,
        background_neighb_grid_0,
        search_template,
        K=2e5,
        G=2e5,
        port_pos_0="elastic_sph.pos_0",
        port_pos_now="basic.pos",
        port_mass="basic.mass",
        port_vel="basic.vel",
        port_volume="basic.rest_volume",
        port_L="elastic_sph.L",
        port_F="elastic_sph.F",
        port_R="elastic_sph.R",
        port_P="elastic_sph.P",
        port_eps="elastic_sph.eps",
    ):
        required_capacity = [
            "node_basic",
            "node_sph",
            "node_ISPH_Elastic",
            "node_implicit_sph",
        ]
        for capacity in required_capacity:
            if not capacity in obj.capacity_list:
                print("Exception for the construction of ISPH_Elastic():")
                print("obj does not have the capacity " + capacity)
                exit(0)

        self.obj = obj
        self.dt = dt
        self.background_neighb_grid = background_neighb_grid
        self.background_neighb_grid_0 = background_neighb_grid_0
        self.search_template = search_template

        self.obj_pos_0 = eval("self.obj." + (port_pos_0))
        self.obj_pos_now = eval("self.obj." + (port_pos_now))
        self.obj_mass = eval("self.obj." + (port_mass))
        self.obj_vel = eval("self.obj." + (port_vel))
        self.obj_volume = eval("self.obj." + (port_volume))
        self.obj_L = eval("self.obj." + (port_L))
        self.obj_F = eval("self.obj." + (port_F))
        self.obj_R = eval("self.obj." + (port_R))
        self.obj_P = eval("self.obj." + (port_P))
        self.obj_eps = eval("self.obj." + (port_eps))

        self.K = ti.field(ti.f32, ())
        self.G = ti.field(ti.f32, ())
        self.K[None] = float(K)
        self.G[None] = float(G)

        self.background_neighb_grid_0.register(self.obj_pos_now)
        self.obj.attr_set_arr(self.obj_pos_0, self.obj_pos_now)

        self.kernel_init()

        self.obj.clear(self.obj.elastic_sph.L)
        self.compute_L(
            obj_volume=self.obj_volume,
            obj_pos_0=self.obj_pos_0,
            obj_output_L=self.obj_L,
            background_neighb_grid=self.background_neighb_grid_0,
            search_template=self.search_template,
        )
        pass

    # Eqn.3
    @ti.kernel
    def compute_F(
        self,
        obj: ti.template(),
        obj_volume: ti.template(),
        obj_pos_0: ti.template(),
        obj_pos_now: ti.template(),
        obj_L: ti.template(),
        obj_output_F: ti.template(),
        background_neighb_grid: ti.template(),
        search_template: ti.template(),
    ):
        for pid in range(obj.info.stack_top[None]):
            located_cell = background_neighb_grid.get_located_cell(
                pos=obj_pos_0[pid],
            )
            for neighb_cell_iter in range(search_template.get_neighb_cell_num()):
                neighb_cell_index = background_neighb_grid.get_neighb_cell_index(
                    located_cell=located_cell,
                    cell_iter=neighb_cell_iter,
                    neighb_search_template=search_template,
                )
                if background_neighb_grid.within_grid(neighb_cell_index):
                    for neighb_part in range(
                        background_neighb_grid.get_cell_part_num(neighb_cell_index)
                    ):
                        nid = background_neighb_grid.get_neighb_part_id(
                            cell_index=neighb_cell_index,
                            neighb_part_index=neighb_part,
                        )
                        # compute down below
                        x_ji_0 = obj_pos_0[nid] - obj_pos_0[pid]
                        dis_0 = x_ji_0.norm()
                        if dis_0 > 1e-6:
                            x_ji_now = obj_pos_now[nid] - obj_pos_now[pid]
                            grad_W_vec = obj_L[pid] @ (
                                grad_spline_W(
                                    dis_0, obj.sph.h[pid], obj.sph.sig_inv_h[pid]
                                )
                                * (-x_ji_0)
                                / dis_0
                            )
                            obj_output_F[pid] += obj_volume[
                                nid
                            ] * x_ji_now.outer_product(grad_W_vec)

    # Eqn.1
    def compute_L(
        self,
        obj_volume,
        obj_pos_0,
        obj_output_L,
        background_neighb_grid,
        search_template,
    ):
        self.compute_L_ker(
            self.obj,
            obj_volume,
            obj_pos_0,
            obj_output_L,
            background_neighb_grid,
            search_template,
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
        background_neighb_grid: ti.template(),
        search_template: ti.template(),
    ):
        for pid in range(obj.info.stack_top[None]):
            located_cell = background_neighb_grid.get_located_cell(
                pos=obj_pos_0[pid],
            )
            for neighb_cell_iter in range(search_template.get_neighb_cell_num()):
                neighb_cell_index = background_neighb_grid.get_neighb_cell_index(
                    located_cell=located_cell,
                    cell_iter=neighb_cell_iter,
                    neighb_search_template=search_template,
                )
                if background_neighb_grid.within_grid(neighb_cell_index):
                    for neighb_part in range(
                        background_neighb_grid.get_cell_part_num(neighb_cell_index)
                    ):
                        nid = background_neighb_grid.get_neighb_part_id(
                            cell_index=neighb_cell_index,
                            neighb_part_index=neighb_part,
                        )
                        # compute down below
                        x_ji_0 = obj_pos_0[nid] - obj_pos_0[pid]
                        dis_0 = x_ji_0.norm()
                        if dis_0 > 1e-6:
                            grad_W_vec = (
                                grad_spline_W(
                                    dis_0, obj.sph.h[pid], obj.sph.sig_inv_h[pid]
                                )
                                * (-x_ji_0)
                                / dis_0
                            )
                            obj_output_L[pid] += obj_volume[
                                nid
                            ] * grad_W_vec.outer_product(x_ji_0)

    @ti.kernel
    def compute_R_pd(
        self,
        obj: ti.template(),
        obj_F: ti.template(),
        obj_output_R: ti.template(),
    ):
        for pid in range(obj.info.stack_top[None]):
            obj_output_R[pid] = ti.polar_decompose(obj_F[pid])[0]

    # Eqn.5
    @ti.kernel
    def compute_F_star(
        self,
        obj: ti.template(),
        obj_volume: ti.template(),
        obj_pos_0: ti.template(),
        obj_pos_now: ti.template(),
        obj_R: ti.template(),
        obj_L: ti.template(),
        obj_output_F_star: ti.template(),
        background_neighb_grid: ti.template(),
        search_template: ti.template(),
    ):
        dim = ti.static(obj.basic.pos[0].n)
        I = ti.Matrix.identity(dt=ti.f32, n=dim)
        for pid in range(obj.info.stack_top[None]):
            located_cell = background_neighb_grid.get_located_cell(
                pos=obj_pos_0[pid],
            )
            for neighb_cell_iter in range(search_template.get_neighb_cell_num()):
                neighb_cell_index = background_neighb_grid.get_neighb_cell_index(
                    located_cell=located_cell,
                    cell_iter=neighb_cell_iter,
                    neighb_search_template=search_template,
                )
                if background_neighb_grid.within_grid(neighb_cell_index):
                    for neighb_part in range(
                        background_neighb_grid.get_cell_part_num(neighb_cell_index)
                    ):
                        nid = background_neighb_grid.get_neighb_part_id(
                            cell_index=neighb_cell_index,
                            neighb_part_index=neighb_part,
                        )
                        # compute down below
                        x_ji_0 = obj_pos_0[nid] - obj_pos_0[pid]
                        dis_0 = x_ji_0.norm()
                        if dis_0 > 1e-6:
                            x_ji_now = obj_pos_now[nid] - obj_pos_now[pid]
                            grad_W_vec = (
                                obj_R[pid]
                                @ obj_L[pid]
                                @ (
                                    grad_spline_W(
                                        dis_0, obj.sph.h[pid], obj.sph.sig_inv_h[pid]
                                    )
                                    * (-x_ji_0)
                                    / dis_0
                                )
                            )
                            obj_output_F_star[pid] += obj_volume[nid] * (
                                x_ji_now - (obj_R[pid] @ x_ji_0)
                            ).outer_product(grad_W_vec)
            obj_output_F_star[pid] += I

    @ti.kernel
    def compute_eps(
        self,
        obj: ti.template(),
        obj_F: ti.template(),
        obj_output_eps: ti.template(),
    ):
        I = ti.Matrix.identity(dt=ti.f32, n=obj_F[0].n)
        for pid in range(obj.info.stack_top[None]):
            obj_output_eps[pid] = (obj_F[pid] + obj_F[pid].transpose()) * 0.5 - I

    @ti.kernel
    def compute_P(
        self,
        obj: ti.template(),
        obj_eps: ti.template(),
        obj_output_P: ti.template(),
    ):
        I = ti.Matrix.identity(dt=ti.f32, n=obj_eps[0].n)
        for pid in range(obj.info.stack_top[None]):
            obj_output_P[pid] = (2 * self.G[None] * obj_eps[pid]) + (
                (self.K[None] - (2 / 3 * self.G[None])) * obj_eps[pid].trace() * I
            )

    @ti.kernel
    def compute_force(
        self,
        obj: ti.template(),
        obj_volume: ti.template(),
        obj_pos_0: ti.template(),
        obj_R: ti.template(),
        obj_L: ti.template(),
        obj_P: ti.template(),
        obj_output_force: ti.template(),
        background_neighb_grid: ti.template(),
        search_template: ti.template(),
    ):
        for pid in range(obj.info.stack_top[None]):
            located_cell = background_neighb_grid.get_located_cell(
                pos=obj_pos_0[pid],
            )
            for neighb_cell_iter in range(search_template.get_neighb_cell_num()):
                neighb_cell_index = background_neighb_grid.get_neighb_cell_index(
                    located_cell=located_cell,
                    cell_iter=neighb_cell_iter,
                    neighb_search_template=search_template,
                )
                if background_neighb_grid.within_grid(neighb_cell_index):
                    for neighb_part in range(
                        background_neighb_grid.get_cell_part_num(neighb_cell_index)
                    ):
                        nid = background_neighb_grid.get_neighb_part_id(
                            cell_index=neighb_cell_index,
                            neighb_part_index=neighb_part,
                        )
                        # compute down below
                        x_ji_0 = obj_pos_0[nid] - obj_pos_0[pid]
                        dis_0 = x_ji_0.norm()
                        if dis_0 > 1e-6:
                            grad_W_vec_i = (
                                obj_R[pid]
                                @ obj_L[pid]
                                @ (
                                    grad_spline_W(
                                        dis_0, obj.sph.h[pid], obj.sph.sig_inv_h[pid]
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
                                        dis_0, obj.sph.h[nid], obj.sph.sig_inv_h[nid]
                                    )
                                    * (x_ji_0)
                                    / dis_0
                                )
                            )
                            obj_output_force[pid] += (
                                obj_volume[pid]
                                * obj_volume[nid]
                                * (
                                    (obj_P[pid] @ grad_W_vec_i)
                                    - (obj_P[nid] @ grad_W_vec_nid)
                                )
                            )

    def internal_loop(self, output_force):
        self.obj.clear(self.obj_F)
        self.compute_F(
            obj=self.obj,
            obj_volume=self.obj_volume,
            obj_pos_0=self.obj_pos_0,
            obj_pos_now=self.obj_pos_now,
            obj_L=self.obj_L,
            obj_output_F=self.obj_F,
            background_neighb_grid=self.background_neighb_grid_0,
            search_template=self.search_template,
        )

        self.compute_R_pd(
            obj=self.obj,
            obj_F=self.obj_F,
            obj_output_R=self.obj_R,
        )

        # / compute F_star with F cleared first /
        self.obj.clear(self.obj_F)
        self.compute_F_star(
            obj=self.obj,
            obj_volume=self.obj_volume,
            obj_pos_0=self.obj_pos_0,
            obj_pos_now=self.obj_pos_now,
            obj_R=self.obj_R,
            obj_L=self.obj_L,
            obj_output_F_star=self.obj_F,
            background_neighb_grid=self.background_neighb_grid_0,
            search_template=self.search_template,
        )
        # / compute epsilon /
        self.compute_eps(
            obj=self.obj,
            obj_F=self.obj_F,
            obj_output_eps=self.obj_eps,
        )

        # / compute P /
        self.compute_P(
            obj=self.obj,
            obj_eps=self.obj_eps,
            obj_output_P=self.obj_P,
        )

        # / compute force /
        self.compute_force(
            obj=self.obj,
            obj_volume=self.obj_volume,
            obj_pos_0=self.obj_pos_0,
            obj_R=self.obj_R,
            obj_L=self.obj_L,
            obj_P=self.obj_P,
            obj_output_force=output_force,
            background_neighb_grid=self.background_neighb_grid_0,
            search_template=self.search_template,
        )

    # / update acc /
    @ti.kernel
    def update_acc(
        self,
        input_force: ti.template(),
        output_acc: ti.template(),
    ):
        for pid in range(self.obj.info.stack_top[None]):
            output_acc[pid] += input_force[pid] / self.obj_mass[pid]

    def compute_vis(
        self,
        kinetic_vis_coeff: ti.template(),
        output_acc: ti.template(),
    ):
        self.compute_Laplacian(
            obj=self.obj,
            obj_pos=self.obj_pos_now,
            nobj_pos=self.obj_pos_now,
            nobj_volume=self.obj_volume,
            obj_input_attr=self.obj_vel,
            nobj_input_attr=self.obj_vel,
            coeff=kinetic_vis_coeff,
            obj_output_attr=output_acc,
            background_neighb_grid=self.background_neighb_grid,
            search_template=self.search_template,
        )
