import taichi as ti
from taichi.lang.ops import atomic_min


@ti.kernel
def set_value_to_attr_1d(num: ti.i32, attr: ti.template(), val: ti.template()):
    for i in range(num):
        attr[i] = val


@ti.kernel
def set_value_to_attr_2d(num_i: ti.i32, attr: ti.template(), val: ti.template()):
    for i in range(num_i):
        for j in ti.static(range(attr.n)):
            attr[i][j] = val


@ti.kernel
def copy_value_to_attr(
    num: ti.i32, input_attr: ti.template(), output_attr: ti.template()
):
    for i in range(num):
        output_attr[i] = input_attr[i]


@ti.kernel
def cfl_condition_dt(
    num: ti.i32,
    vel: ti.template(),
    discretization_size: ti.f32,
    cfl_factor: ti.f32,
    max_dt: ti.f32,
    min_divisor: ti.f32,
) -> ti.f32:
    output_dt = max_dt
    for i in range(num):
        vel_norm = vel[i].norm()
        if vel_norm > min_divisor:
            atomic_min(output_dt, discretization_size / vel_norm * cfl_factor)
    return output_dt
