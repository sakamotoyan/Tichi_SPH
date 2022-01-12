from sph import *

class Scenario:
    scenario_buffer = None
    scenario_wait_buffer = [] # scenario objects waiting to be added at a certain time
    inlets = [] #stores inlets that pour water in
    fluid = None #reference to fluid obj
    bound = None #reference to bound obj

    def __init__(self, scenario_file_path):
        self.scenario_buffer = get_scenario_buffer(trim_path_dir(scenario_file_path))

    def get_obj_from_str(self, s):
        if s == 'fluid':
            return self.fluid
        elif s == 'bound':
            return self.bound
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


def add_to_scenario(obj_str, param, scenario, config):
    obj = scenario.get_obj_from_str(obj_str)
    if obj is None:
        print('scenario WARNING: obj \'',obj_str,'\' non-exist.')
    else:
        if param['type']=='inlet':
            if np.linalg.norm(np.array(param['norm'])) < 1e-6:
                raise Exception('inlet ERROR: magnitute of attribute \'norm\' should not be zero.')
            param['obj']=obj_str
            param['dt']=inlet_helper_get_dt(param['norm'],param['speed'],param['particle_relaxing_factor'],config)
            param['t_pre']=config.time_count[None]-param['dt']
            scenario.inlets.append(param)
        else:
            obj.push_scene_obj(param, config)

def refresh_inlets(scenario, config):
    for i in range(len(scenario.inlets)-1,-1,-1):
        inlet=scenario.inlets[i]
        if 'end_time' in inlet and config.time_count[None]>=inlet['end_time']:
            scenario.inlets.pop(i)
        elif config.time_count[None]>=inlet['t_pre']+inlet['dt']:
            try:
                inlet['t_pre']=config.time_count[None]
                scenario.get_obj_from_str(inlet['obj']).scene_add_from_inlet(
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
                scenario.inlets.pop(i)

def refresh_scenario(scenario, config):
    for i in range(len(scenario.scenario_wait_buffer)-1,-1,-1):
        param=scenario.scenario_wait_buffer[i]
        try:
            if param['start_time'] <= config.time_count[None]:
                add_to_scenario(param['_part'], param, scenario, config)
                scenario.scenario_wait_buffer.pop(i)
        except Exception:
            warn('scenario WARNING: invalid scenario object:',json.dumps(param))
            scenario.scenario_wait_buffer.pop(i)
    refresh_inlets(scenario, config)
    SPH_update_color(scenario.fluid,config)

def init_scenario(fluid, bound, scenario, config):
    scenario.fluid=fluid
    scenario.bound=bound
    """ setup scene """
    try:
        for part in scenario.scenario_buffer:
            if part != "sim_env":
                for param in scenario.scenario_buffer[part]['objs']:
                    if not 'start_time' in param or param['start_time'] <= 0.0: # add now
                        add_to_scenario(part, param, scenario, config)
                    else: # add later
                        param['_part']=part
                        scenario.scenario_wait_buffer.append(param)
    except Exception:
        raise Exception('scenario ERROR: no scenario file or scenario file invalid.')
    set_unused_par(fluid,config)
    set_unused_par(bound,config)
    SPH_update_color(fluid,config)