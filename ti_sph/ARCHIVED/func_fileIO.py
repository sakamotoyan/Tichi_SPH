import os
import json
import numpy as np
from plyfile import *

# Input:  original_file_path->string
# Output: trimmed_file_path->string
def trim_path_dir(original_file_path):
    if original_file_path.find('\\') > 0 and original_file_path.find('/') > 0:
        return original_file_path
    elif original_file_path.find('\\') > 0:
        file_path_list = original_file_path.split('\\')
    elif original_file_path.find('/') > 0:
        file_path_list = original_file_path.split('/')
    trimmed_file_path = file_path_list[0]
    for i in range(len(file_path_list)-1):
        trimmed_file_path = os.path.join(trimmed_file_path, file_path_list[i+1])
    return trimmed_file_path


# Input:  path-> String
# Output: verts_array-> numpy array (num*dim) dtype=float32  
def read_ply(path):
    obj_ply = PlyData.read(path)
    obj_verts = obj_ply['vertex'].data
    verts_array = np.array([[x, y, z] for x, y, z in obj_verts])
    return verts_array


def write_ply(path, frame_num, dim, num, pos):
    if dim == 3:
        list_pos = [(pos[i, 0], pos[i, 1], pos[i, 2]) for i in range(num)]
    elif dim == 2:
        list_pos = [(pos[i, 0], pos[i, 1], 0) for i in range(num)]
    else:
        print('write_ply(): dim exceeds default values')
        return
    np_pos = np.array(list_pos, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el_pos = PlyElement.describe(np_pos, 'vertex')
    PlyData([el_pos]).write(str(path) + '_' + str(frame_num) + '.ply')

def get_scenario_buffer(scenario_file_path):
    try:
        scenario_buffer = json.load(open(scenario_file_path))
        return scenario_buffer
    except Exception:
        print('Error from sph_config.py: no scenario file or scenario file invalid')
        exit()

def push_part_from_ply(self, scenario_buffer, obj_name, config):
        for obj in scenario_buffer:
            if (obj == obj_name):
                for param in scenario_buffer[obj]['objs']:
                    if param['type'] == 'cube':
                        self.scene_add_cube(param['start_pos'], param['end_pos'], param['volume_frac'], param['vel'],
                                            int(param['color'], 16), param['particle_relaxing_factor'], config)
                    elif param['type'] == 'box':
                        self.scene_add_box(param['start_pos'], param['end_pos'], param['layers'], param['volume_frac'],
                                            param['vel'], int(param['color'], 16), param['particle_relaxing_factor'], config)
                    elif param['type'] == 'ply':
                        verts = read_ply(trim_path_dir(param['file_name']))
                        self.push_part_seq(len(verts), int(param['color'], 16), verts, ti.Vector(param['volume_frac']), ti.Vector(param['vel']),
                                                config)

def get_config_buffer(config_file_path):
    try:
        config_buffer = json.load(open(config_file_path))
        return config_buffer
    except Exception:
        print('Error from sph_config.py: no config file or config file invalid')
        exit()