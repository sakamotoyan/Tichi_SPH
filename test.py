import taichi as ti
import ti_sph as tsph
from ti_sph.gui import pos_normalizer, ti2numpy_color


ti.init()

a = ti.field(ti.f32, ())
b = ti.field(ti.f32, ())

a[None] = 1.5
b[None] = 2

print(tsph.tmp_func(a[None], b[None]))

num = ti.field(ti.i32, ())
num[None] = 24

particle_field = ti.Struct.field({
    "pos": ti.types.vector(3, ti.f32),
    "color": ti.f32,
    "pos_normalized": ti.types.vector(3, ti.f32),
    "vel": ti.types.vector(3, ti.f32),
    "acc": ti.types.vector(3, ti.f32),
    "mass": ti.f32,
}, shape=(num[None],))

space_lb = ti.Vector.field(3, ti.f32, shape=())
space_rt = ti.Vector.field(3, ti.f32, shape=())
space_lb_tmp = ti.Vector([-1.0,-1.0,-1.0])
space_rt_tmp = ti.Vector([1.0,1.0,1.0])

space_lb[None] = space_lb_tmp
space_rt[None] = space_rt_tmp

# pos_normalizer(num, particle_field.pos, space_lb, space_rt, particle_field.pos_normalized)
color_numpy = ti2numpy_color(num[None], particle_field.color)


# print(particle_field.pos_normalized)
print(color_numpy)

