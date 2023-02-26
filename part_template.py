import taichi as ti
from ti_sph.basic_op import *
from ti_sph import *
import numpy as np

def fluid_part_template(part_num, dim=3, verbose=False):
    fluid_part = Particle(part_num)

    fluid_part.add_attr("rest_density", val_f())
    fluid_part.add_attr("color_RGBA", vec4_f())
    ## fluid_part.add_attr("test_attr_list", [0.1, 0.2, 1])

    fluid_part.add_array("pos", vecxf(dim).field())
    fluid_part.add_array("vel", vecxf(dim).field())
    fluid_part.add_array("mass", ti.field(ti.f32))
    ## example only, not used in this test
    # fluid_part.add_array("volume_fraction", [ti.field(ti.f32), ti.field(ti.f32)])

    fluid_phase = ti.types.struct(
    val_frac=ti.f32,
    phase_vel=vecxf(dim),
    phase_acc=vecxf(dim),
    phase_force=vecxf(dim),
    )

    sph = ti.types.struct(
        density=ti.f32,
        pressure=ti.f32,
        pressure_force=vecxf(dim),
        viscosity_force=vecxf(dim),
        gravity_force=vecxf(dim),
    )

    fluid_part.add_struct("phases", [fluid_phase, fluid_phase])
    fluid_part.add_struct("sph", sph)

    if verbose:
        fluid_part.verbose_attrs("fluid_part")
        fluid_part.verbose_arrays("fluid_part")
        fluid_part.verbose_structs("fluid_part")
    
    return fluid_part
