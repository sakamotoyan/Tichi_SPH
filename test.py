import taichi as ti
import ti_sph as tsph
from ti_sph.gui import pos_normalizer, ti2numpy_color
from ti_sph.sph_tools import *


@ti.kernel
def loop1(num1: ti.i32, num2: ti.i32, attr1: ti.template(), attr2: ti.template()):
    for i in range(num1):
        loop2(num2, attr1[i], attr2)


@ti.kernel
def loop2(num: ti.i32, attr1: ti.template(), attr2: ti.template()):
    attr1[i] += 1
    for i in range(num):
        attr2[i] += 1


ti.init()

a = ti.field(ti.f32, ())
b = ti.field(ti.f32, ())

a[None] = 1.5
b[None] = 2

print(tsph.tmp_func(a[None], b[None]))

num = ti.field(ti.i32, ())
dim = ti.field(ti.i32, ())
num[None] = 24
dim[None] = 3

particle_field = ti.Struct.field({
    "pos": ti.types.vector(dim[None], ti.f32),
    "color": ti.f32,
    "pos_normalized": ti.types.vector(3, ti.f32),
    "vel": ti.types.vector(dim[None], ti.f32),
    "acc": ti.types.vector(dim[None], ti.f32),
    "mass": ti.f32,
}, shape=(num[None],))

set_value_to_attr_1d(num[None], particle_field.pos, ti.Vector([1.1, 2.2, 3.3]))
copy_value_to_attr(num[None], particle_field.pos, particle_field.vel)

loop1(num[None], num[None], particle_field.color, particle_field.mass)

print(particle_field.mass)

# space_lb = ti.Vector.field(dim[None], ti.f32, shape=())
# space_rt = ti.Vector.field(dim[None], ti.f32, shape=())
# space_lb_tmp = ti.Vector([-1.0,-1.0,-1.0])
# space_rt_tmp = ti.Vector([1.0,1.0,1.0])
# space_lb[None] = space_lb_tmp
# space_rt[None] = space_rt_tmp

# # pos_normalizer(num, particle_field.pos, space_lb, space_rt, particle_field.pos_normalized)
# color_numpy = ti2numpy_color(num[None], particle_field.color)


# # print(particle_field.pos_normalized)
# print(color_numpy)
