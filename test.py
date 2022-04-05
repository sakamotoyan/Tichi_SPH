import taichi as ti
import ti_sph as tsph
from ti_sph.class_node import test
ti.init()

# CONFIG
config_capacity = ['info_space', 'info_discretization',
                   'info_sim', 'info_neighb_search', 'info_gui']
config = tsph.Config(dim=3, capacity_list=config_capacity)
# space
config_space = ti.static(config.space)
config_space.dim[None] = 3
config_space.lb.fill(-8)
config_space.lb[None][1] = -4
config_space.rt.fill(8)
# discretization
config_discre = ti.static(config.discre)
config_discre.part_size[None] = 0.1
config_discre.cs[None] = 220
config_discre.cfl_factor[None] = 0.5
config_discre.dt[None] = tsph.fixed_dt(
    config_discre.cs[None], config_discre.part_size[None], config_discre.cfl_factor[None])
# neighbour search
config_neighb = ti.static(config.neighb)
config_neighb.cell_size[None] = config_discre.part_size[None] * 2
tsph.calculate_neighb_cell_param(config_neighb, config_space)
# gui
config_gui = ti.static(config.gui)
config_gui.res[None] = [1920, 1080]
config_gui.frame_rate[None] = 60
config_gui.cam_fov[None] = 55
config_gui.cam_pos[None] = [6.0, 1.0, 0.0]
config_gui.cam_look[None] = [0.0, 0.0, 0.0]
config_gui.canvas_color[None] = [0.2, 0.2, 0.6]
config_gui.ambient_light_color[None] = [0.7, 0.7, 0.7]
config_gui.point_light_pos[None] = [2, 1.5, -1.5]
config_gui.point_light_color[None] = [0.8, 0.8, 0.8]


# FLUID
fluid_capacity = ["node_basic", 'node_color',
                  'node_implicit_sph', 'node_neighb_search']
fluid = tsph.Node(dim=config_space.dim[None], id=0, part_num=int(1e5),
                  neighb_cell_num=config_neighb.cell_num[None], capacity_list=fluid_capacity)
fluid.color.color_vec.fill(ti.Vector([1, 1, 0]))
fluid.push_cube(ti.Vector([-1, -1, -1]),
                ti.Vector([1, 1, 1]), config_discre.part_size[None], 1)

# BOUND
bound_capacity = ["node_basic", 'node_color',
                  'node_implicit_sph', 'node_neighb_search']
bound = tsph.Node(dim=config_space.dim[None], id=0, part_num=int(1e5),
                  neighb_cell_num=config_neighb.cell_num[None], capacity_list=bound_capacity)
bound.color.color_vec.fill(ti.Vector([0.5, 0.5, 0.5]))
bound.push_box(ti.Vector([-1.5, -1.5, -1.5]),
               ti.Vector([1.5, 1, 1.5]), config_discre.part_size[None], 1, 2)

fluid.neighb_search(config_neighb, config_space)

test(fluid, 0)

# GUI
# gui = tsph.Gui(config.gui)
# gui.env_set_up()
# while gui.window.running:
#     if gui.op_system_run == True:
#         a = 1
#     gui.monitor_listen()
#     if gui.op_refresh_window:
#         gui.scene_setup()
#         gui.scene_add_parts(fluid, length=config_discre.part_size[None])
#         gui.scene_add_parts(bound, length=config_discre.part_size[None])
#         gui.scene_render()


# print(config.space)
# print(config.discre)
# print(config.neighb)
# print(config.sim)
# print(config.gui)
