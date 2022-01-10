import numpy
from sph import *
import json
import time

time_count=0

'''parse command line'''
config_file_path = 'config/config.json'
scenario_file_path = 'scenario/3d_inlet_demo.json'

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

scenario_wait_buffer = [] # scenario objects waiting to be added at a certain time
inlets = [] #stores inlets that pour water in
def get_obj_from_str(s):
    if s == 'fluid':
        return fluid
    elif s == 'bound':
        return bound
    else:
        print('scenario WARNING: object does not exist (name:',s,')')
        return None

#helper function for inlet
def inlet_helper_get_dt(norm,speed,relaxing_factor,config):
    norm=np_normalize(np.array(norm))
    if speed<0:
        norm, speed = -norm, -speed
    acc_vec=config.gravity.to_numpy()
    acc=np.dot(norm,acc_vec)
    r = config.part_size[1]
    if acc < 1e-6:
        if speed > 0:
            return r * relaxing_factor / speed
        else:
            return float("inf")
    else:
        return ((speed ** 2 + 2 * acc * r) ** 0.5 - speed) / acc * relaxing_factor


def add_to_scenario(obj_str,param):
    obj = get_obj_from_str(obj_str)
    if obj is None:
        print('scenario WARNING: obj \'',obj_str,'\' non-exist.')
    else:
        if param['type'] == 'cube':
            obj.scene_add_cube(param['start_pos'], param['end_pos'], param['volume_frac'], param['vel'], int(param['color'], 16), param['particle_relaxing_factor'],config)
        elif param['type'] == 'box':
            obj.scene_add_box(param['start_pos'], param['end_pos'], param['layers'], param['volume_frac'], param['vel'], int(param['color'], 16), param['particle_relaxing_factor'],config)
        elif param['type'] == 'ply':
            verts = read_ply(param['file_name'])
            obj.scene_add_ply(len(verts), verts, param['volume_frac'], param['vel'], int(param['color'], 16),config)
        elif param['type']=='inlet':
            if np.linalg.norm(np.array(param['norm'])) < 1e-6:
                raise Exception('inlet ERROR: magnitute of attribute \'norm\' should not be zero.')
            param['obj']=obj_str
            param['dt']=inlet_helper_get_dt(param['norm'],param['speed'],param['particle_relaxing_factor'],config)
            param['t_pre']=time_count-param['dt']
            inlets.append(param)

def refresh_inlets():
    for i in range(len(inlets)-1,-1,-1):
        inlet=inlets[i]
        if 'end_time' in inlet and time_count>=inlet['end_time']:
            inlets.pop(i)
        elif time_count>=inlet['t_pre']+inlet['dt']:
            try:
                inlet['t_pre']=time_count
                get_obj_from_str(inlet['obj']).scene_add_from_inlet(
                    inlet['center'],
                    inlet['size'],
                    inlet['norm'],
                    inlet['speed'],
                    inlet['volume_frac'],
                    int(inlet['color'], 16),
                    inlet['particle_relaxing_factor'],
                    config
                )
            except Exception:
                warn('inlet WARNING: refresh_inlets failed for:',json.dumps(inlet))
                inlets.pop(i)

def refresh_scenario():
    for i in range(len(scenario_wait_buffer)-1,-1,-1):
        param=scenario_wait_buffer[i]
        try:
            if param['time'] <= time_count:
                add_to_scenario(param['_part'],param)
                scenario_wait_buffer.pop(i)
        except Exception:
            warn('scenario WARNING: invalid scenario object:',json.dumps(param))
            scenario_wait_buffer.pop(i)
    refresh_inlets()
    SPH_update_color(fluid,config)

def init_scenario():
    """ setup scene """
    try:
        for part in scenario_buffer:
            if part != "sim_env":
                for param in scenario_buffer[part]['objs']:
                    if not 'time' in param or param['time'] <= 0.0: # add now
                        add_to_scenario(part,param)
                    else: #add later
                        param['_part']=part
                        scenario_wait_buffer.append(param)
    except Exception:
        raise Exception('scenario ERROR: no scenario file or scenario file invalid.')
    set_unused_par(fluid,config)
    set_unused_par(bound,config)
    SPH_update_color(fluid,config)

init_scenario()
print('Done pushing particles')

gui = Gui(config)
gui.env_set_up()
print('Done gui setting')

while gui.window.running:
    if gui.op_system_run == True:
        sph_step(ngrid, fluid, bound, config)
        time_count += config.dt[None]
        refresh_scenario()
    gui.monitor_listen()
    if gui.op_refresh_window:
        gui.scene_setup()
        gui.scene_add_objs(obj=fluid, radius=config.part_size[1] * 0.5)
        if gui.show_bound:
            gui.scene_add_objs(obj=bound, radius=config.part_size[1] * 0.5)
        gui.scene_render()