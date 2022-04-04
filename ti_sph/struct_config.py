from turtle import shape
import taichi as ti


def struct_config_space(dim):
    struct_config_space = ti.types.struct(
        dim=ti.i32,
        lb=ti.types.vector(dim, ti.f32),
        rt=ti.types.vector(dim, ti.f32),
    )
    return struct_config_space.field(shape=())


def struct_config_discretization():
    struct_config_discretization = ti.types.struct(
        part_size=ti.f32,
        dt=ti.f32,
        cs=ti.f32,
        cfl_factor=ti.f32,
    )
    return struct_config_discretization.field(shape=())


def struct_config_neighb_search(dim):
    struct_config_neighb_search = ti.types.struct(
        cell_size=ti.f32,
        cell_num=ti.i32,
        cell_num_vec=ti.types.vector(dim, ti.i32),
        cell_coder=ti.types.vector(dim, ti.i32),
    )
    return struct_config_neighb_search.field(shape=())


def struct_config_sim(dim):
    struct_config_sim = ti.types.struct(
        gravity=ti.types.vector(dim, ti.f32),
        fluid_dynamic_vis=ti.f32,
    )
    return struct_config_sim.field(shape=())


def struct_config_gui():
    struct_config_gui = ti.types.struct(
        res=ti.types.vector(2, ti.i32),
        frame_rate=ti.i32,
        
        cam_fov=ti.f32,
        cam_pos=ti.types.vector(3, ti.f32),
        cam_look=ti.types.vector(3, ti.f32),
        
        canvas_color=ti.types.vector(3, ti.f32),
        
        ambient_light_color=ti.types.vector(3, ti.f32),
        point_light_pos=ti.types.vector(3, ti.f32),
        point_light_color=ti.types.vector(3, ti.f32),
    )
    return struct_config_gui.field(shape=())
