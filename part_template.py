import taichi as ti
from ti_sph.basic_op import *
from ti_sph import *
import numpy as np

def part_template(part_obj, verbose=False):
    ''' Enssential arrays'''
    ''' encouraged to add for any particle system'''
    part_obj.add_array("pos", vecxf(part_obj.m_world.g_dim[None]).field())
    part_obj.add_array("vel", vecxf(part_obj.m_world.g_dim[None]).field())
    part_obj.add_array("mass", ti.field(ti.f32))
    part_obj.add_array("size", ti.field(ti.f32))
    part_obj.add_array("volume", ti.field(ti.f32))
    part_obj.add_array("rest_density", ti.field(ti.f32))
    part_obj.add_array("acc", vecxf(part_obj.m_world.g_dim[None]).field())

    ''' Optional arrays'''
    part_obj.add_array("volume_fraction", ti.field(ti.f32), bundle=2)
    part_obj.add_array("vel_adv", vecxf(part_obj.m_world.g_dim[None]).field())
    ## example only, not used in this test
    # fluid_part.add_array("volume_fraction", [ti.field(ti.f32), ti.field(ti.f32)])

    fluid_phase = ti.types.struct(
    val_frac=ti.f32,
    phase_vel=vecxf(part_obj.m_world.g_dim[None]),
    phase_acc=vecxf(part_obj.m_world.g_dim[None]),
    phase_force=vecxf(part_obj.m_world.g_dim[None]),
    )

    sph = ti.types.struct(
        h=ti.f32,
        sig=ti.f32,
        sig_inv_h=ti.f32,
        density=ti.f32,
        pressure=ti.f32,
        pressure_force=vecxf(part_obj.m_world.g_dim[None]),
        viscosity_force=vecxf(part_obj.m_world.g_dim[None]),
        gravity_force=vecxf(part_obj.m_world.g_dim[None]),
    )

    sph_df = ti.types.struct(
        alpha_1=vecxf(part_obj.m_world.g_dim[None]),
        alpha_2=ti.f32,
        alpha=ti.f32,
        delta_density=ti.f32,
        vel_adv=vecxf(part_obj.m_world.g_dim[None]),
    )

    part_obj.add_struct("phases", fluid_phase, bundle=2)
    part_obj.add_struct("sph", sph)
    part_obj.add_struct("sph_df", sph_df)

    if verbose:
        part_obj.verbose_attrs("fluid_part")
        part_obj.verbose_arrays("fluid_part")
        part_obj.verbose_structs("fluid_part")

    return part_obj
