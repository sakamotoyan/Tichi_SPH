import numpy
from sph import *
import json
import time
import ti_sph as tsph

'''parse command line'''
config_file_path = 'config/config.json'
# scenario_file_path = 'scenario/3d_dambreak_2block.json'
scenario_file_path = 'scenario/3d_dambreak.json'

""" init data structure """
config_buffer = get_config_buffer(trim_path_dir(config_file_path))
scenario_buffer = get_scenario_buffer(trim_path_dir(scenario_file_path))
print('Done reading files')

taichi_init(config_buffer)
pre_config = Pre_config(config_buffer, scenario_buffer)
config = Config(pre_config, config_buffer, scenario_buffer)
print('Done configurating')

ngrid = Ngrid(config)
fluid = Fluid(config.fluid_max_part_num[None], pre_config, config)
bound = Fluid(config.bound_max_part_num[None], pre_config, config)
print('Done instancing')

fluid.push_part_from_ply(scenario_buffer, 'fluid', config)
bound.push_part_from_ply(scenario_buffer, 'bound', config)
print('Done pushing particles')

gui = tsph.Gui(config)
gui.env_set_up()
print('Done gui setting')

while gui.window.running:
    if gui.op_system_run == True:
        sph_step(ngrid, fluid, bound, config)
    gui.monitor_listen()
    if gui.op_refresh_window:
        gui.scene_setup()
        gui.scene_add_objs(obj=fluid, radius=config.part_size[1] * 0.5)
        if gui.show_bound:
            gui.scene_add_objs(obj=bound, radius=config.part_size[1] * 0.5)
        gui.scene_render()