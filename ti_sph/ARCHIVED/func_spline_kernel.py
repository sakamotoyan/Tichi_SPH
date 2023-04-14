# seems duplicate with SPH_kernel, it seems SPH_kernel contains the actually used functions
# SPH kernels

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
        tmp = (2 * (1 - q) ** 3 * q ** 3 - 1 / 64)
    elif q > 0.5 and q < 1:
        tmp = (1 - q) ** 3 * q ** 3
    tmp *= 32 / math.pi / h ** 3
    return tmp

# FROM: Eqn.(4) of the paper "Smoothed Particle Hydrodynamics Techniques for the Physics Based Simulation of Fluids and Solids"
# REF: https://github.com/InteractiveComputerGraphics/SPH-Tutorial/blob/master/pdf/SPH_Tutorial.pdf
# NOTE: $dim$ is implicitly defined in the param $sig$
@ti.func
def spline_W(r, h, sig):
    q = r / h
    tmp = 0.0
    if q <= 0.5:
        tmp = 6 * (q ** 3 - q ** 2) + 1
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
        tmp = 6 * (3 * q ** 2 - 2 * q)
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
def artificial_Laplacian_spline_W(r, h, sig, dim, V_j, x_ij: ti.template(), A_ij: ti.template()):
    return 2 * (2 + dim) * V_j * grad_spline_W(r, h, sig) * x_ij.normalized() * A_ij.dot(x_ij) / (r ** 2)