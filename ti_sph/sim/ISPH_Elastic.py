# PAPER:    An Implicit SPH Formulation for Incompressible Linearly Elastic Solids
# AUTHOR:   Andreas Peer, Christoph Gissler, Stefan Band, Matthias Teschner
# JOURNAL:  COMPUTER GRAPHICS FORUM Volume37, Issue6
# URL:      https://onlinelibrary.wiley.com/doi/10.1111/cgf.13317

import taichi as ti
from .SPH_kernel import *


@ti.data_oriented
class ISPH_Elastic:
    def __init__(self) -> None:
        pass

    def compute_deformation_gradient(
        self,
        obj: ti.template(),
        obj_volume: ti.template(),
        obj_output_attr: ti.template(),
        config_neighb: ti.template(),
    ):
        self.compute_deformation_gradient_k(
            obj,
            obj_volume,
            obj_output_attr,
            config_neighb,
        )

    @ti.kernel
    def compute_deformation_gradient_k(
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
                            obj_output_attr[i] += obj_volume[nid] * x_ji.outer_product(
                                grad_W_vec
                            )
