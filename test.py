import taichi as ti
from ti_sph.basic_op import data_op
from ti_sph import *
import numpy as np

ti.init()

# arr_a = ti.Vector.field(3, ti.f32, (5))
# arr_b = ti.Vector.field(3, ti.f32, (5))


# ker_arr_fill(arr_a, 50, 0, arr_a.shape[0])
# ker_arr_cpy(from_arr=arr_a, to_arr=arr_b, num=arr_a.shape[0], offset=vec2i(0))
# ker_arr_fill(val=3.3, to_arr=arr_b, num=1, offset=1)
# ker_arr_set(to_arr=arr_b, val=ti.Vector([1.1, 2.2, 3.3]), num=1, offset=2)
# ker_arr_add(to_arr=arr_b, val=1.1, num=1, offset=4)


# print(arr_b)

part_num = 10

struct_node_basic = ti.types.struct(
    pos=ti.types.vector(3, ti.f32),   # position
    vel=ti.types.vector(3, ti.f32),   # velocity
    acc=ti.types.vector(3, ti.f32),   # acceleration (not always used)
    force=ti.types.vector(3, ti.f32),  # force (not always used)
    mass=ti.f32,                        # mass
    rest_density=ti.f32,                # rest density
    rest_volume=ti.f32,                 # rest volume
    size=ti.f32,                        # diameter
)

struct_node_basic2 = ti.types.struct(
    pos=ti.types.vector(3, ti.f32),   # position
    vel=ti.types.vector(3, ti.f32),   # velocity
    acc=ti.types.vector(3, ti.f32),   # acceleration (not always used)
    force=ti.types.vector(3, ti.f32),  # force (not always used)
)

struct_list = [struct_node_basic, struct_node_basic2]

fluid_part = Particle(part_num)
fluid_part.add_array("gr", ti.field(ti.f32))
fluid_part.add_array("gr2", vec3f.field())
fluid_part.add_arrays("phases", [ti.field(ti.f32),ti.field(ti.i32),vec3f.field()])

fluid_part.add_struct("struct_node_basic", struct_node_basic)
fluid_part.add_structs("phases2", [struct_node_basic2, struct_node_basic2])

fluid_part.verbose_arrays("fluid_part arrays:")
fluid_part.verbose_structs("fluid_part structs:")

ker_arr_fill(val=3.3, to_arr=fluid_part.gr2, num=1, offset=1)
ker_arr_set(to_arr=fluid_part.phases[2], val=ti.Vector([1.1, 2.2, 3.3]), num=1, offset=2)
ker_arr_fill(to_arr=fluid_part.gr2, val=10, num=1, offset=0)
ker_arr_set(to_arr=fluid_part.gr, val=10, num=1, offset=2)
# print(fluid_part.phases[0])

cube_gen = Cube_generator(lb=vec3f(0,1,3), rt=vec3f(0.3,1.3,3.2), span=0.1)
print(cube_gen.generate_pos_based_on_span())

dim = [10, 10, 10]
# a 10x10x10 numpy array with each element as a 3D vector)

