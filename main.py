import numpy
from scenario import *
import json
import time

'''parse command line'''
config_file_path = 'config/config.json'
scenario_file_path = 'scenario/3d_double_dambreak.json'

""" init data structure """
config_buffer = get_config_buffer(trim_path_dir(config_file_path))
scenario = Scenario(scenario_file_path)
print('Done reading files')

taichi_init(config_buffer)
pre_config = Pre_config(config_buffer, scenario.scenario_buffer)
config = Config(pre_config, config_buffer, scenario.scenario_buffer)
print('Done configurating')

ngrid = Ngrid(config)
fluid = Fluid(config.fluid_max_part_num[None], pre_config, config)
bound = Fluid(config.bound_max_part_num[None], pre_config, config)
print('Done instancing')

init_scenario(fluid, bound, scenario, config)
print('Done pushing particles')

gui = Gui(config)
gui.env_set_up()
print('Done gui setting')

# start_id, end_id = bound.get_part_range_from_name('rod')
# vel_down_np = np.array([0.0,-3.0,0.0])
# vel_rot_np = np.zeros(3)
# ang_spd = math.pi
# rot_r = 0.5
# time_down = 0.5

# rod_vel = ti.Vector.field(config.dim[None],float,())
# def apply_bound_transform(bound, config):
#     bound.update_pos_part_range(start_id, end_id, config)
#     if config.time_count[None] < time_down:
#         rod_vel.from_numpy(vel_down_np)
#     else:
#         ang = ang_spd * (config.time_count[None] - time_down)
#         vel_rot_np[0] = ang_spd * rot_r * cos(ang)
#         vel_rot_np[2] = ang_spd * rot_r * sin(ang)
#         rod_vel.from_numpy(vel_rot_np)
#     bound.set_vel_part_range(start_id, end_id, rod_vel)

while gui.window.running:
    if gui.op_system_run == True:
        sph_step(ngrid, fluid, bound, config)
        config.time_count[None] += config.dt[None]
        refresh_scenario(scenario, config)
        # apply_bound_transform(bound, config)
        # gui.op_system_run = False
    gui.monitor_listen()
    if gui.op_refresh_window:
        gui.scene_setup()
        fluid.display_all()
        gui.scene_add_objs(obj=fluid, radius=config.part_size[1] * 0.5)
        if gui.show_bound:
            bound.display_all()
            gui.scene_add_objs(obj=bound, radius=config.part_size[1] * 0.5)
        # elif gui.show_rod:
        #     bound.display_part_range(start_id,end_id)
        #     gui.scene_add_objs(obj=bound, radius=config.part_size[1] * 0.5)
        gui.scene_render()
    if gui.show_stat:
        print(config.time_count[None])
        print(fluid.volume_frac_tmp[0])
        gui.show_stat = False