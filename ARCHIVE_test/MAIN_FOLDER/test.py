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

# STEP 2.1: init particle data structure
part_num_fluid = val_i(1e4)
part_num_bound = val_i(1e4)

fluid_part = Particle(part_num_fluid[None])
bound_part = Particle(part_num_bound[None])

# STEP 2.2: add attributes to particle data structure
fluid_part.add_attr("rest_density", val_f())
# example only, not used in this test
fluid_part.add_attr("test_attr_list", [0.1, 0.2, 1])
fluid_part.add_attr("color_RGBA", vec4_f(0))

# STEP 2.3: add arrays to particle data structure
fluid_part.add_array("pos", vec3f.field())
fluid_part.add_array("vel", vec3f.field())
fluid_part.add_array("mass", ti.field(ti.f32))
# example only, not used in this test
fluid_part.add_array("volume_fraction", [ti.field(ti.f32), ti.field(ti.f32)])

# STEP 2.4: add structs to particle data structure
part_neighb_search = ti.types.struct(
    prev=ti.i32,
    next=ti.i32,
)
fluid_phase = ti.types.struct(
    val_frac=ti.f32,
    phase_vel=ti.types.vector(3, ti.f32),
    phase_acc=ti.types.vector(3, ti.f32),
    phase_force=vec3f,
)
sph = ti.types.struct(
    density=ti.f32,
    pressure=ti.f32,
    pressure_force=vec3f,
    viscosity_force=vec3f,
    gravity_force=vec3f,
)
fluid_part.add_struct("neighb_search", part_neighb_search)
fluid_part.add_struct("phases", [fluid_phase, fluid_phase])
fluid_part.add_struct("sph", sph)

# STEP 2.5: verbose particle data structure
fluid_part.verbose_attrs("fluid_part")
fluid_part.verbose_arrays("fluid_part")
fluid_part.verbose_structs("fluid_part")

# STEP 2.6: same to the bound part
bound_part.add_attr("rest_density", val_f())
bound_part.add_attr("color_RGBA", vec4f(0))
bound_part.add_array("pos", vec3f.field())
bound_part.add_array("vel", vec3f.field())
bound_part.add_array("mass", ti.field(ti.f32))
bound_part.add_struct("sph", sph)
bound_part.verbose_attrs("bound_part")
bound_part.verbose_arrays("bound_part")
bound_part.verbose_structs("bound_part")

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
fluid_part.set_from_val(fluid_part.mass, pushed_num[None],
                        g_part_size[None]**3 * fluid_part.rest_density[None])
fluid_part.update_stack_top(pushed_num[None])
# END of STEP 3.1: generate data for fluid part

# STEP 3.2: generate data for bound part
cube_gen = Cube_generator(lb=vec3f(0, 0, 3), rt=vec3f(0.3, 0.2, 3.2))
cube_gen.generate_pos_based_on_span(span=g_part_size[None])
pushed_num = val_i(cube_gen.num)
bound_part.set_from_numpy(bound_part.pos, cube_gen.pos_arr)
bound_part.set_from_val(bound_part.mass, pushed_num[None],
                        g_part_size[None]**3 * bound_part.rest_density[None])
bound_part.update_stack_top(pushed_num[None])
# END of STEP 3.2: generate data for bound part

print("bound_part.mass: ", bound_part.mass.to_numpy()[:pushed_num[None]])

# ------------------------- END of STEP 3 ------------------------------

bound_NS = Neighb_search_FS(dim=3, cell_size=g_supporrt_radius[None], lb=g_simspace_lb,
                            rt=g_simspace_rt, obj=bound_part, obj_pos=bound_part.pos)