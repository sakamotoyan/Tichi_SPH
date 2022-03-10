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
def tmp_func(a:ti.f32, b:ti.f32)->ti.i32:
    return int(ti.ceil(a/b))