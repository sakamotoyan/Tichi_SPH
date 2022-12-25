import taichi as ti
from .type import *
""" Monocular """


@ti.kernel
def ker_arr_fill(
    to_arr: ti.template(),
    val: ti.template(),
    offset: ti.i32,
    num: ti.i32,
):
    for i in range(offset, offset + num):
        for n in ti.static(range(to_arr.n)):
            to_arr[i][n] = val


@ti.kernel
def ker_arr_set(
    to_arr: ti.template(),
    val: ti.template(),
    offset: ti.i32,
    num: ti.i32,
):
    for i in range(offset, offset + num):
        to_arr[i] = val


@ti.kernel
def ker_arr_add(
    to_arr: ti.template(),
    val: ti.template(),
    offset: ti.i32,
    num: ti.i32,
):
    for i in range(offset, offset + num):
        to_arr[i] += val

""" Binocular """


@ti.kernel
def ker_arr_cpy(
    to_arr: ti.template(),
    from_arr: ti.template(),
    offset: vec2i,  # offset[0] for to_arr, offset[1] for from_arr
    num: ti.i32,
):
    arr_n = ti.static(from_arr.n)
    for i in range(num):
        for n in ti.static(range(arr_n)):
            to_arr[i+offset[0]][n] = from_arr[i+offset[1]][n]
