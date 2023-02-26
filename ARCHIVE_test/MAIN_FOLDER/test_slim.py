import taichi as ti
from ti_sph.basic_op import *
from ti_sph import *
import numpy as np
from SIM_SCENE.part_template import fluid_part_template


# STEP 0: init taichi
ti.init()

# ------------------------- END of STEP 0 ------------------------------

# STEP 1: define global parameters
g_part_size = val_f(0.1)  # particle size
g_supporrt_radius = val_f(2*g_part_size[None])  # support radius
g_gravity = vec3_f([0, -9.8, 0])  # gravity
g_dt = val_f(1e-3)  # time step
g_rest_density = val_f(1000)  # rest density
g_kinematic_viscosity = val_f(1e-3)  # kinematic viscosity
g_simspace_lb = vec3f(-8, -8, -8)  # simulation space lower bound
g_simspace_rt = vec3f(8, 8, 8)  # simulation space upper bound

# ------------------------- END of STEP 1 ------------------------------

# STEP 2: define particle data structure

part_num_fluid = val_i(1e4)
part_num_bound = val_i(1e4)

fluid_part = fluid_part_template(part_num_fluid[None])
bound_part = fluid_part_template(part_num_bound[None])

# ------------------------- END of STEP 2 ------------------------------

# STEP 3: init data value
fluid_part.rest_density.fill(1000)
fluid_part.color_RGBA.fill(vec4f([1, 0, 0, 1]))
bound_part.rest_density.fill(1000)
bound_part.color_RGBA.fill(vec4f([0, 0, 1, 1]))


# STEP 3.1: Cube_generator to generate numpy pos array for particles
cube_gen = Cube_generator(lb=vec3f(0, 1, 3), rt=vec3f(0.3, 1.3, 3.2))
cube_gen.generate_pos_based_on_span(span=g_part_size[None])
pushed_num = val_i(cube_gen.num)
fluid_part.set_from_numpy(fluid_part.pos, cube_gen.pos_arr)
fluid_part.set_from_val(fluid_part.mass, pushed_num[None], g_part_size[None]**3 * fluid_part.rest_density[None])
fluid_part.update_stack_top(pushed_num[None])
# END of STEP 3.1: generate data for fluid part


# STEP 3.2: generate data for bound part
cube_gen = Cube_generator(lb=vec3f(0, 0, 3), rt=vec3f(0.3, 0.2, 3.2))
cube_gen.generate_pos_based_on_span(span=g_part_size[None])
pushed_num = val_i(cube_gen.num)
bound_part.set_from_numpy(bound_part.pos, cube_gen.pos_arr)
bound_part.set_from_val(bound_part.mass, pushed_num[None], g_part_size[None]**3 * bound_part.rest_density[None])
bound_part.update_stack_top(pushed_num[None])
# END of STEP 3.2: generate data for bound part

# ------------------------- END of STEP 3 ------------------------------

bound_NS = Neighb_search_FS(dim=3, cell_size=g_supporrt_radius[None], lb=g_simspace_lb,
                            rt=g_simspace_rt, obj=bound_part, obj_pos=bound_part.pos)