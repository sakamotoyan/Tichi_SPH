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