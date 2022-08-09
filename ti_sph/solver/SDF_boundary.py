import taichi as ti

# for SDF boundary handling
@ti.func
def lamb(d, h):
    q = abs(d) / h
    tmp = 0.0
    if q <= 0.5:
        tmp = 1.0 / 60 * (192 * q ** 6 - 288 * q ** 5 + 160 * q ** 3 - 84 * q + 30)
    elif q > 0.5 and q < 1:
        tmp = -8.0 / 15 * (2 * q ** 6 - 9 * q ** 5 + 15 * q ** 4 - 10 * q ** 3 + 3 * q - 1)
    if d < 0:
        tmp = 1 - tmp
    return tmp

# for SDF boundary handling
@ti.func
def lamb_grad_norm(d, h):
    q = abs(d) / h
    tmp = 0.0
    if q <= 0.5:
        tmp = 1.0 / 5 * (96 * q ** 5 - 120 * q ** 4 + 40 * q ** 2 - 7)
    elif q > 0.5 and q < 1:
        tmp = -8.0 / 5 * (4 * q ** 5 - 15 * q ** 4 + 20 * q ** 3 - 10 * q ** 2 + 1)
    # if d < 0: #paper seems wrong?
    #     tmp = -tmp
    return tmp

# sdf contribution with penalty
@ti.func
def sdf_contrib(pos_i, h, from_sdf):
    sdf = from_sdf.sdf(pos_i)
    beta = 1.0 - sdf / h
    return beta * lamb(sdf, h)

# sdf grad contribution with penalty
@ti.func
def sdf_grad_contrib(pos_i, h, from_sdf):
    sdf = from_sdf.sdf(pos_i)
    beta = 1.0 - sdf / h
    d_beta = -1.0 / h
    sdf_grad = from_sdf.sdf_grad(pos_i)
    lamb_grad = sdf_grad * d_beta * lamb(sdf, h) + beta * sdf_grad / h * lamb_grad_norm(sdf, h)
    return lamb_grad
