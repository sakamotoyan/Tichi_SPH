# defines structs used in Config

from turtle import shape
import taichi as ti

# siulation space
# info_construct()
# "info_space" -> space
def struct_config_space(dim):
    struct_config_space = ti.types.struct(
        dim=ti.i32,                         # dimensions (e.g. 2D, 3D)
        lb=ti.types.vector(dim, ti.f32),    # simulation space limit (corner with smallest coordinate in each dimension)
        rt=ti.types.vector(dim, ti.f32),    # simulation space limit (corner with largest coordinate in each dimension)
    )
    return struct_config_space.field(shape=())


# space and time discretization
# info_construct()
# "info_discretization" -> discre
def struct_config_discretization():
    struct_config_discretization = ti.types.struct(
        part_size=ti.f32,   # particle diameter
        dt=ti.f32,          # time step length
        inv_dt=ti.f32,      # 1/dt
        cs=ti.f32,          # speed of sound (for cfl etc.)
        cfl_factor=ti.f32,  # cfl condition factor
    )
    return struct_config_discretization.field(shape=())


# not used? (related info is already in Neighb_Cell)
# info_construct()
# "info_neighb_search" -> neighb
def struct_config_neighb_search(dim):
    struct_config_neighb_search = ti.types.struct(
        cell_size=ti.f32,
        cell_num=ti.i32,
        cell_num_vec=ti.types.vector(dim, ti.i32),
        cell_coder=ti.types.vector(dim, ti.i32),
        search_range=ti.i32,
    )
    return struct_config_neighb_search.field(shape=())


# simulation parameters
# info_construct()
# "info_sim" -> sim
def struct_config_sim(dim):
    struct_config_sim = ti.types.struct(
        gravity=ti.types.vector(dim, ti.f32),   # gravity
        dynamic_vis=ti.f32,                     # dynamic viscosity (not used, doesn't seem neccessary given kinematic_vis)
        kinematic_vis=ti.f32,                   # kinematic viscosity
    )
    return struct_config_sim.field(shape=())


# gui parameters
# info_construct()
# "info_gui" -> gui
def struct_config_gui():
    struct_config_gui = ti.types.struct(
        res=ti.types.vector(2, ti.i32),                 # resolution
        frame_rate=ti.i32,                              # frame rate
        cam_fov=ti.f32,                                 # camera FOV (field of view)
        cam_pos=ti.types.vector(3, ti.f32),             # position of camera
        cam_look=ti.types.vector(3, ti.f32),            # camera lookat position
        canvas_color=ti.types.vector(3, ti.f32),        # background color
        ambient_light_color=ti.types.vector(3, ti.f32), # ambient light color
        point_light_pos=ti.types.vector(3, ti.f32),     # point light position
        point_light_color=ti.types.vector(3, ti.f32),   # point light color
    )
    return struct_config_gui.field(shape=())
