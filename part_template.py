import taichi as ti
from ti_sph.basic_op import *
from ti_sph import *
import numpy as np

def part_template(part_num, world: World, verbose=False):
    fluid_part = Particle(part_num, world, world.part_size)

    ''' Enssential arrays'''
    ''' encouraged to add for any particle system'''
    fluid_part.add_array("pos", vecxf(world.dim[None]).field())
    fluid_part.add_array("vel", vecxf(world.dim[None]).field())
    fluid_part.add_array("mass", ti.field(ti.f32))
    fluid_part.add_array("size", ti.field(ti.f32))
    fluid_part.add_array("volume", ti.field(ti.f32))
    fluid_part.add_array("rest_density", ti.field(ti.f32))
    fluid_part.add_array("acc", vecxf(world.dim[None]).field())

    ''' Optional arrays'''
    fluid_part.add_array("volume_fraction", ti.field(ti.f32), bundle=2)
    fluid_part.add_array("vel_adv", vecxf(world.dim[None]).field())
    ## example only, not used in this test
    # fluid_part.add_array("volume_fraction", [ti.field(ti.f32), ti.field(ti.f32)])

    fluid_phase = ti.types.struct(
    val_frac=ti.f32,
    phase_vel=vecxf(world.dim[None]),
    phase_acc=vecxf(world.dim[None]),
    phase_force=vecxf(world.dim[None]),
    )

    sph = ti.types.struct(
        h=ti.f32,
        sig=ti.f32,
        sig_inv_h=ti.f32,
        density=ti.f32,
        pressure=ti.f32,
        pressure_force=vecxf(world.dim[None]),
        viscosity_force=vecxf(world.dim[None]),
        gravity_force=vecxf(world.dim[None]),
    )

    sph_df = ti.types.struct(
        alpha_1=vecxf(world.dim[None]),
        alpha_2=ti.f32,
        alpha=ti.f32,
        delta_density=ti.f32,
        vel_adv=vecxf(world.dim[None]),
    )

    fluid_part.add_struct("phases", fluid_phase, bundle=2)
    fluid_part.add_struct("sph", sph)
    fluid_part.add_struct("sph_df", sph_df)

    if verbose:
        fluid_part.verbose_attrs("fluid_part")
        fluid_part.verbose_arrays("fluid_part")
        fluid_part.verbose_structs("fluid_part")

    return fluid_part
