import taichi as ti
from ti_sph.basic_op import *
from ti_sph import *
import numpy as np

def part_template(part_num, dim=3, verbose=False):
    fluid_part = Particle(part_num)

    fluid_part.add_attr("color_RGBA_", vec4_f())
    ## fluid_part.add_attr("test_attr_list", [0.1, 0.2, 1])

    fluid_part.add_array("pos_", vecxf(dim).field())
    fluid_part.add_array("vel_", vecxf(dim).field())
    fluid_part.add_array("acc_", vecxf(dim).field())
    fluid_part.add_array("mass_", ti.field(ti.f32))
    fluid_part.add_array("size_", ti.field(ti.f32))
    fluid_part.add_array("volume_", ti.field(ti.f32))
    fluid_part.add_array("rest_density_", ti.field(ti.f32))
    fluid_part.add_array("volume_fraction_", ti.field(ti.f32), bundle=2)
    fluid_part.add_array("vel_adv_", vecxf(dim).field())
    ## example only, not used in this test
    # fluid_part.add_array("volume_fraction", [ti.field(ti.f32), ti.field(ti.f32)])

    fluid_phase = ti.types.struct(
    val_frac=ti.f32,
    phase_vel=vecxf(dim),
    phase_acc=vecxf(dim),
    phase_force=vecxf(dim),
    )

    sph = ti.types.struct(
        h=ti.f32,
        sig=ti.f32,
        sig_inv_h=ti.f32,
        density=ti.f32,
        pressure=ti.f32,
        pressure_force=vecxf(dim),
        viscosity_force=vecxf(dim),
        gravity_force=vecxf(dim),
    )

    sph_df = ti.types.struct(
        alpha_1=vecxf(dim),
        alpha_2=ti.f32,
        alpha=ti.f32,
        delta_density=ti.f32,
        vel_adv=vecxf(dim),
    )

    fluid_part.add_struct("phases_", fluid_phase, bundle=2)
    fluid_part.add_struct("sph_", sph)
    fluid_part.add_struct("sph_df_", sph_df)

    if verbose:
        fluid_part.verbose_attrs("fluid_part")
        fluid_part.verbose_arrays("fluid_part")
        fluid_part.verbose_structs("fluid_part")

    return fluid_part
