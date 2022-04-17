import taichi as ti
import math

from ti_sph.func_util import distance_1, distance_2

# FROM: Eqn.(2) of the paper "Versatile Surface Tension and Adhesion for SPH Fluids"
# REF: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.462.8293&rep=rep1&type=pdf
# NOTE: this func is insensitive to the $dim$
@ti.func
def spline_C(r, h):
    q = r / h
    tmp = 0.0
    if q <= 0.5:
        tmp = 2 * (1 - q) ** 3 * q**3 - 1 / 64
    elif q > 0.5 and q < 1:
        tmp = (1 - q) ** 3 * q**3
    tmp *= 32 / math.pi / h**3
    return tmp


# FROM: Eqn.(4) of the paper "Smoothed Particle Hydrodynamics Techniques for the Physics Based Simulation of Fluids and Solids"
# REF: https://github.com/InteractiveComputerGraphics/SPH-Tutorial/blob/master/pdf/SPH_Tutorial.pdf
# NOTE: $dim$ is implicitly defined in the param $sig$
@ti.func
def spline_W(r, h, sig):
    q = r / h
    tmp = 0.0
    if q <= 0.5:
        tmp = 6 * (q**3 - q**2) + 1
    elif q > 0.5 and q < 1:
        tmp = 2 * (1 - q) ** 3
    tmp *= sig
    return tmp


# FROM: Eqn.(4) of the paper "Smoothed Particle Hydrodynamics Techniques for the Physics Based Simulation of Fluids and Solids"
# REF: https://github.com/InteractiveComputerGraphics/SPH-Tutorial/blob/master/pdf/SPH_Tutorial.pdf
# NOTE: This fun is spline_W() with the derivative of $r$
@ti.func
def grad_spline_W(r, h, sig):
    q = r / h
    tmp = 0.0
    if q <= 0.5:
        tmp = 6 * (3 * q**2 - 2 * q)
    elif q > 0.5 and q < 1:
        tmp = -6 * (1 - q) ** 2
    tmp *= sig / h
    return tmp


# FROM: Eqn.(26) of the paper "Smoothed Particle Hydrodynamics Techniques for the Physics Based Simulation of Fluids and Solids"
# REF: https://github.com/InteractiveComputerGraphics/SPH-Tutorial/blob/master/pdf/SPH_Tutorial.pdf
# NOTE: x_ij and A_ij should be all Vector and be alinged
#       e.g. x_ij=ti.Vector([1,2]) A_ij=ti.Vector([1,2])
# NOTE: V_j is the volume of particle j, V_j==m_j/rho_j==Vj0/compression_rate_j
@ti.func
def artificial_Laplacian_spline_W(
    r, h, sig, dim, V_j, x_ij: ti.template(), A_ij: ti.template()
):
    return (
        2
        * (2 + dim)
        * V_j
        * grad_spline_W(r, h, sig)
        * x_ij.normalized()
        * A_ij.dot(x_ij)
        / (r**2)
    )


@ti.data_oriented
class SPH_kernel:
    def __init__(self):
        pass

    def compute_sig(self, obj):
        if not "node_sph" in obj.capacity_list:
            print("Exception from compute_sig():")
            print("obj has no capacity ' node_sph '")
            exit(0)
        dim = ti.static(obj.basic.pos[0].n)
        sig = 0
        if dim == 3:
            sig = 8 / math.pi
        elif dim == 2:
            sig = 40 / 7 / math.pi
        elif dim == 1:
            sig = 4 / 3
        else:
            print("Exception from compute_sig_k():")
            print("dim out of range")
            exit(0)
        self.compute_sig_k(obj, sig)

    def compute_W_arr(
        self,
        obj,
        obj_output_attr,
        nobj,
        nobj_volume,
        nobj_input_attr,
        config_neighb,
    ):
        self.compute_W_arr_k(
            obj, obj_output_attr, nobj, nobj_volume, nobj_input_attr, config_neighb
        )

    def compute_W_const(
        self,
        obj,
        obj_output_attr,
        nobj,
        nobj_volume,
        nobj_input_attr,
        config_neighb,
    ):
        self.compute_W_const_k(
            obj, obj_output_attr, nobj, nobj_volume, nobj_input_attr, config_neighb
        )

    @ti.kernel
    def compute_sig_k(self, obj: ti.template(), sig: ti.f32):
        dim = ti.static(obj.basic.pos[0].n)
        for i in range(obj.info.stack_top[None]):
            obj.sph.sig[i] = sig / ti.pow(obj.sph.h[i], dim)

    @ti.kernel
    def set_h(self, obj: ti.template(), h: ti.f32):
        dim = ti.static(obj.basic.pos[0].n)
        for i in range(obj.info.stack_top[None]):
            obj.sph.h[i] = h

    @ti.kernel
    def compute_W_arr_k(
        self,
        obj: ti.template(),
        obj_output_attr: ti.template(),
        nobj: ti.template(),
        nobj_volume: ti.template(),
        nobj_input_attr: ti.template(),
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
                        dis = distance_1(obj.basic.pos[i], nobj.basic.pos[nid])
                        obj_output_attr[i] += (
                            nobj_input_attr[nid]
                            * nobj_volume[nid]
                            * spline_W(dis, obj.sph.h[i], obj.sph.sig[i])
                        )

    @ti.kernel
    def compute_W_const_k(
        self,
        obj: ti.template(),
        obj_output_attr: ti.template(),
        nobj: ti.template(),
        nobj_volume: ti.template(),
        nobj_input_attr: ti.template(),
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
                        dis = distance_1(obj.basic.pos[i], nobj.basic.pos[nid])
                        obj_output_attr[i] += (
                            nobj_input_attr
                            * nobj_volume[nid]
                            * spline_W(dis, obj.sph.h[i], obj.sph.sig[i])
                        )
