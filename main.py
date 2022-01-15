import numpy
from scenario import *
import json
import time

'''parse command line'''
config_file_path = 'config/config.json'
scenario_file_path = 'scenario/3d_rotate_rod.json'

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

config.start_id[None], config.end_id[None] = bound.get_part_range_from_name('rod')
config.vel_down_np = np.array([0.0,-3.0,0.0])
config.vel_rot_np = np.zeros(3)

while gui.window.running:
    if gui.op_system_run == True:
        run_step(ngrid, fluid, bound, config)
        refresh_scenario(scenario, config)
    gui.monitor_listen()
    if gui.op_refresh_window:
        gui.scene_setup()
        fluid.display_all()
        gui.scene_add_objs(obj=fluid, radius=config.part_size[1] * 0.5)
        if gui.show_bound:
            bound.display_all()
            gui.scene_add_objs(obj=bound, radius=config.part_size[1] * 0.5)
        elif gui.show_rod:
            bound.display_part_range(config.start_id[None], config.end_id[None])
            gui.scene_add_objs(obj=bound, radius=config.part_size[1] * 0.5)
        gui.scene_render()
        if gui.op_write_file:
            write_files(gui, config, pre_config, fluid)
        gui.window_show()

