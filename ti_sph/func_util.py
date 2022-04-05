import taichi as ti
import os


@ti.func
def if_array_has_negative(vec: ti.template()):
    is_n = False
    for i in ti.static(range(vec.n)):
        if vec[i] < 0:
            is_n = True
    return is_n


@ti.kernel
def tmp_func(a: ti.f32, b: ti.f32) -> ti.i32:
    return int(ti.ceil(a/b))


@ti.func
def has_negative(vec) -> bool:
    has_negative = False
    for i in ti.static(range(vec.n)):
        if vec[i] < 0:
            has_negative = True
    return has_negative


@ti.func
def has_positive(vec) -> bool:
    has_positive = False
    for i in ti.static(range(vec.n)):
        if vec[i] > 0:
            has_positive = True
    return has_positive


@ti.func
def node_encode(node_pos: ti.template(), lb: ti.template(), cell_size: ti.f32):
    return int((node_pos - lb[None]) // cell_size)