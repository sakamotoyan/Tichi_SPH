import taichi as ti
from ti_sph.basic_op import data_op
from ti_sph import *
import numpy as np

# STEP 0: init taichi
ti.init()

# ------------------------- END of STEP 0 ------------------------------

# STEP 1: define global parameters
g_part_size = 0.1 # particle size
g_supporrt_radius = 2 * g_part_size # support radius
g_gravity = vec3f(0, -9.8, 0) # gravity
g_dt = 1e-3 # time step
g_rest_density = 1000 # rest density
g_kinematic_viscosity = 1e-3 # kinematic viscosity

# ------------------------- END of STEP 1 ------------------------------

# STEP 2: define particle data structure

# STEP 2.1: init particle data structure
part_num_fluid = int(1e4)
part_num_bound = int(1e4)

fluid_part = Particle(part_num_fluid)
bound_part = Particle(part_num_bound)

# STEP 2.2: add attributes to particle data structure
fluid_part.add_attr("rest_density", 1000)
fluid_part.add_attr("test_attr_list", [0.1,0.2,1]) # example only, not used in this test
fluid_part.add_attr("color_RGBA", vec4f([1,0,0,1]))

# STEP 2.3: add arrays to particle data structure
fluid_part.add_array("pos", vec3f.field())
fluid_part.add_array("vel", vec3f.field())
fluid_part.add_array("mass", ti.field(ti.f32))
fluid_part.add_array("volume_fraction", [ti.field(ti.f32),ti.field(ti.f32)]) # example only, not used in this test

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
# fluid_part.verbose_attrs("fluid_part")
# fluid_part.verbose_arrays("fluid_part")
# fluid_part.verbose_structs("fluid_part")

# STEP 2.6: same to the bound part
bound_part.add_attr("rest_density", 1000)
bound_part.add_attr("color_RGBA", vec4f([0,0,1,1]))
bound_part.add_array("pos", vec3f.field())
bound_part.add_array("vel", vec3f.field())
bound_part.add_array("mass", ti.field(ti.f32))
bound_part.add_struct("sph", sph)
# bound_part.verbose_attrs("bound_part")
# bound_part.verbose_arrays("bound_part")
# bound_part.verbose_structs("bound_part")

# ------------------------- END of STEP 2 ------------------------------

# STEP 3: init data value

# STEP 3.1: Cube_generator to generate numpy pos array for particles
cube_gen = Cube_generator(lb=vec3f(0,1,3), rt=vec3f(0.3,1.3,3.2))
cube_gen.generate_pos_based_on_span(span=g_part_size)
num = cube_gen.num

fluid_part.from_numpy(fluid_part.pos, cube_gen.pos_arr)

fluid_part.update_stack_top(num)
# fluid_part.mass.from_numpy(np.ones(cube_gen.num) * g_part_size**3 * g_rest_density)


# dim = [10, 10, 10]

