# PAPER:    An Implicit SPH Formulation for Incompressible Linearly Elastic Solids
# AUTHOR:   Andreas Peer, Christoph Gissler, Stefan Band, Matthias Teschner
# JOURNAL:  COMPUTER GRAPHICS FORUM Volume37, Issue6
# URL:      https://onlinelibrary.wiley.com/doi/10.1111/cgf.13317

import taichi as ti
from .SPH_kernel import *
import numpy as np


@ti.data_oriented
class ISPH_Elastic:
    def __init__(self) -> None:
        pass

    def compute_F(
        self,
        obj,
        obj_volume,
        pos_0,
        pos_now,
        ker_correct_mat,
        obj_output_attr,
        config_neighb,
    ):
        self.compute_F_ker(
            obj,
            obj_volume,
            pos_0,
            pos_now,
            ker_correct_mat,
            obj_output_attr,
            config_neighb,
        )

    def compute_L(
        self,
        obj,
        obj_volume,
        obj_output_attr,
        config_neighb,
    ):
        self.compute_L_ker(
            obj,
            obj_volume,
            obj_output_attr,
            config_neighb,
        )
        L = obj_output_attr.to_numpy()
        inv = np.linalg.pinv(L[: obj.info.stack_top[None]])
        L[: obj.info.stack_top[None]] = inv
        obj_output_attr.from_numpy(L)

    @ti.kernel
    def compute_F_ker(
        self,
        obj: ti.template(),
        obj_volume: ti.template(),
        pos_0: ti.template(),
        pos_now: ti.template(),
        ker_correct_mat: ti.template(),
        obj_output_attr: ti.template(),
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
                            x_ji_0 = pos_0[nid] - pos_0[i]
                            x_ji_now = pos_now[nid] - pos_now[i]
                            dis_0 = distance_2(x_ji_0)
                            grad_W_vec = ker_correct_mat[i] @ (
                                grad_spline_W(dis_0, obj.sph.h[i], obj.sph.sig[i])
                                * (-x_ji_0)
                                / dis_0
                            )
                            obj_output_attr[i] += obj_volume[
                                nid
                            ] * x_ji_now.outer_product(grad_W_vec)

    @ti.kernel
    def compute_L_ker(
        self,
        obj: ti.template(),
        obj_volume: ti.template(),
        obj_output_attr: ti.template(),
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
                            x_ji = obj.basic.pos[nid] - obj.basic.pos[i]
                            dis = distance_2(x_ji)
                            grad_W_vec = (
                                grad_spline_W(dis, obj.sph.h[i], obj.sph.sig[i])
                                * (-x_ji)
                                / dis
                            )
                            obj_output_attr[i] += obj_volume[
                                nid
                            ] * grad_W_vec.outer_product(x_ji)
