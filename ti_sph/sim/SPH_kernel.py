from numpy import float32
import taichi as ti
import math

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
def spline_W_old(r, h, sig):
    q = r / h
    tmp = 0.0
    if q <= 0.5:
        tmp = 6 * (q**3 - q**2) + 1
    elif q > 0.5 and q < 1:
        tmp = 2 * (1 - q) ** 3
    tmp *= sig
    return tmp


@ti.func
def spline_W(r, h, sig):
    q = r / h
    tmp = 0.0
    if q > 1:
        pass
    elif q > 0.5:
        tmp = 2 * (1 - q) ** 3
        tmp *= sig
    else:
        tmp = 6 * (q**3 - q**2) + 1
        tmp *= sig
    return tmp


# FROM: Eqn.(4) of the paper "Smoothed Particle Hydrodynamics Techniques for the Physics Based Simulation of Fluids and Solids"
# REF: https://github.com/InteractiveComputerGraphics/SPH-Tutorial/blob/master/pdf/SPH_Tutorial.pdf
# NOTE: This fun is spline_W() with the derivative of $r$
@ti.func
def grad_spline_W_old(r, h, sig):
    q = r / h
    tmp = 0.0
    if q <= 0.5:
        tmp = 6 * (3 * q**2 - 2 * q)
    elif q > 0.5 and q < 1:
        tmp = -6 * (1 - q) ** 2
    tmp *= sig / h
    return tmp


@ti.func
def grad_spline_W(r, h, sig_inv_h):
    q = r / h
    tmp = 0.0
    if q > 1:
        pass
    elif q > 0.5:
        tmp = -6 * (1 - q) ** 2
        tmp *= sig_inv_h
    else:
        tmp = 6 * (3 * q**2 - 2 * q)
        tmp *= sig_inv_h
    return tmp


# FROM: Eqn.(26) of the paper "Smoothed Particle Hydrodynamics Techniques for the Physics Based Simulation of Fluids and Solids"
# REF: https://github.com/InteractiveComputerGraphics/SPH-Tutorial/blob/master/pdf/SPH_Tutorial.pdf
# NOTE: x_ij and A_ij should be all Vector and be alinged
#       e.g. x_ij=ti.Vector([1,2]) A_ij=ti.Vector([1,2])
# NOTE: V_j is the volume of particle j, V_j==m_j/rho_j==Vj0/compression_rate_j
@ti.func
def artificial_Laplacian_spline_W(
    r, grad_W, dim, V_j, x_ij: ti.template(), A_ij: ti.template()
):
    return 2 * (2 + dim) * V_j * grad_W * (x_ij) * A_ij.dot(x_ij) / (r**3)


@ti.func
def bigger_than_zero(val: ti.template()):
    if_bigger_than_zero = False
    if val > 1e-6:
        if_bigger_than_zero = True
    return if_bigger_than_zero


@ti.func
def make_bigger_than_zero():
    return 1e-6


@ti.kernel
def fixed_dt(cs: ti.f32, discretization_size: ti.f32, cfl_factor: ti.f32) -> ti.f32:
    return discretization_size / cs * cfl_factor


@ti.kernel
def cfl_dt(
    obj: ti.template(),
    obj_size: ti.template(),
    obj_vel: ti.template(),
    cfl_factor: ti.template(),
    min_acc_norm: ti.f32,
    output_dt: ti.template(),
    output_inv_dt: ti.template(),
):
    dt = ti.Vector([100.0])
    dt[0] = 100

    for i in range(obj.info.stack_top[None]):
        acc_dt = ti.sqrt(obj_size[i] * cfl_factor[None] / min_acc_norm)

        vel_dt = obj_size[i] / 1 * cfl_factor[None]
        vel_norm = obj_vel[i].norm()
        if bigger_than_zero(vel_norm):
            vel_dt = obj_size[i] / vel_norm * cfl_factor[None]

        ti.atomic_min(dt[0], ti.min(vel_dt, acc_dt))

    output_dt[None] = float(dt[0])
    output_inv_dt[None] = 1 / output_dt[None]


@ti.data_oriented
class SPH_kernel:
    def __init__(self):
        pass

    @ti.kernel
    def compute_W(
        self,
        obj: ti.template(),
        nobj: ti.template(),
        nobj_volume: ti.template(),
        nobj_input_attr: ti.template(),
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
                    for j in range(nobj.cell.part_count[cell_coded]):
                        shift = nobj.cell.part_shift[cell_coded] + j
                        nid = nobj.located_cell.part_log[shift]
                        """compute below"""
                        dis = (obj.basic.pos[i], nobj.basic.pos[nid]).norm()
                        obj_output_attr[i] += (
                            nobj_input_attr[nid]
                            * nobj_volume[nid]
                            * spline_W(dis, obj.sph.h[i], obj.sph.sig[i])
                        )

    @ti.kernel
    def compute_W_grad(
        self,
        obj: ti.template(),
        nobj: ti.template(),
        nobj_volume: ti.template(),
        nobj_input_attr: ti.template(),
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
                    for j in range(nobj.cell.part_count[cell_coded]):
                        shift = nobj.cell.part_shift[cell_coded] + j
                        nid = nobj.located_cell.part_log[shift]
                        """compute below"""
                        x_ij = obj.basic.pos[i] - nobj.basic.pos[nid]
                        dis = x_ij.norm()
                        if dis > 1e-6:
                            grad_W_vec = (
                                grad_spline_W(dis, obj.sph.h[i], obj.sph.sig_inv_h[i])
                                * x_ij
                                / dis
                            )
                            obj_output_attr[i] += (
                                nobj_input_attr[nid] * nobj_volume[nid] * grad_W_vec
                            )

    @ti.kernel
    def compute_W_grand_diff(
        self,
        obj: ti.template(),
        nobj: ti.template(),
        nobj_volume: ti.template(),
        obj_input_attr: ti.template(),
        nobj_input_attr: ti.template(),
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
                    for j in range(nobj.cell.part_count[cell_coded]):
                        shift = nobj.cell.part_shift[cell_coded] + j
                        nid = nobj.located_cell.part_log[shift]
                        """compute below"""
                        x_ij = obj.basic.pos[i] - nobj.basic.pos[nid]
                        dis = x_ij.norm()
                        if dis > 1e-6:
                            grad_W_vec = (
                                grad_spline_W(dis, obj.sph.h[i], obj.sph.sig_inv_h[i])
                                * x_ij
                                / dis
                            )
                            obj_output_attr[i] += (
                                (nobj_input_attr[nid] - obj_input_attr[i])
                                * nobj_volume[nid]
                                * grad_W_vec
                            )

    @ti.kernel
    def compute_W_grand_sum(
        self,
        obj: ti.template(),
        nobj: ti.template(),
        nobj_volume: ti.template(),
        obj_input_attr: ti.template(),
        nobj_input_attr: ti.template(),
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
                    for j in range(nobj.cell.part_count[cell_coded]):
                        shift = nobj.cell.part_shift[cell_coded] + j
                        nid = nobj.located_cell.part_log[shift]
                        """compute below"""
                        x_ij = obj.basic.pos[i] - nobj.basic.pos[nid]
                        dis = x_ij.norm()
                        if dis > 1e-6:
                            grad_W_vec = (
                                grad_spline_W(dis, obj.sph.h[i], obj.sph.sig_inv_h[i])
                                * x_ij
                                / dis
                            )
                            obj_output_attr[i] += (
                                (nobj_input_attr[nid] + obj_input_attr[i])
                                * nobj_volume[nid]
                                * grad_W_vec
                            )

    @ti.kernel
    def compute_Laplacian(
        self,
        obj: ti.template(),
        obj_pos: ti.template(),
        nobj: ti.template(),
        nobj_pos: ti.template(),
        nobj_volume: ti.template(),
        obj_input_attr: ti.template(),
        nobj_input_attr: ti.template(),
        coeff: ti.template(),
        obj_output_attr: ti.template(),
        config_neighb: ti.template(),
    ):
        dim = ti.static(obj.basic.pos[0].n)
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
                            A_ij = obj_input_attr[i] - nobj_input_attr[nid]
                            obj_output_attr[i] += coeff[None] * artificial_Laplacian_spline_W(
                                dis,
                                grad_W,
                                dim,
                                nobj_volume[nid],
                                x_ij,
                                A_ij,
                            )

    @ti.kernel
    def set_h(
        self,
        obj: ti.template(),
        obj_output_h: ti.template(),
        h: ti.f32,
    ):
        for i in range(obj.info.stack_top[None]):
            obj_output_h[i] = h

    def compute_sig(self, obj, obj_output_sig):
        dim = ti.static(obj.basic.pos[0].n)
        sig = 0
        if dim == 3:
            sig = 8 / math.pi
        elif dim == 2:
            sig = 40 / 7 / math.pi
        elif dim == 1:
            sig = 4 / 3
        else:
            print("Exception from compute_sig():")
            print("dim out of range")
            exit(0)
        self.compute_sig_ker(obj, obj_output_sig, sig)

    @ti.kernel
    def compute_sig_ker(
        self, obj: ti.template(), obj_output_sig: ti.template(), sig: ti.f32
    ):
        dim = ti.static(obj.basic.pos[0].n)
        for i in range(obj.info.stack_top[None]):
            obj_output_sig[i] = sig / ti.pow(obj.sph.h[i], dim)

    @ti.kernel
    def compute_sig_inv_h(
        self,
        obj: ti.template(),
        obj_sig: ti.template(),
        obj_h: ti.template(),
        obj_output_sig_inv_h: ti.template(),
    ):
        for i in range(obj.info.stack_top[None]):
            obj_output_sig_inv_h[i] = obj_sig[i] / obj_h[i]

    def compute_kernel(
        self,
        obj,
        h,
        obj_output_h,
        obj_output_sig,
        obj_output_sig_inv_h,
    ):
        self.set_h(obj, obj_output_h, h)
        self.compute_sig(obj, obj_output_sig)
        self.compute_sig_inv_h(obj, obj_output_sig, obj_output_h, obj_output_sig_inv_h)
