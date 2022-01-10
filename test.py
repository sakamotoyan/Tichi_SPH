import os
import numpy as np
import taichi as ti
from plyfile import *
ti.init()

# a = ti.Vector.field(3, ti.f32, (10,4))

class TestClass():
    def __init__(self, val):
        self.a = val

def pf1(obj, attr):
    print(attr)

a = ti.Vector([1.1,2.2,3.3])
b=ti.Vector.field(3,ti.f32,())
b[None]=a
print(a[None][0])

# Input:  path-> String
# Output: verts_array-> num*dim size numpy float Array 
def read_ply(path):
    obj_ply = PlyData.read(path)
    obj_verts = obj_ply['vertex'].data
    verts_array = np.array([[x, y, z] for x, y, z in obj_verts])
    return verts_array

a = read_ply('ply_models\\bunny_0.05.ply')

# phase_num = ti.static(config.phase_rest_density.n)
# dim = ti.static(config.gravity.n)

# def trim_path_dir(original_file_path):
#     if original_file_path.find('\\') > 0 and original_file_path.find('/') > 0:
#         return original_file_path
#     elif original_file_path.find('\\') > 0:
#         file_path_list = original_file_path.split('\\')
#     elif original_file_path.find('/') > 0:
#         file_path_list = original_file_path.split('/')
#     trimmed_file_path = file_path_list[0]
#     for i in range(len(file_path_list)-1):
#         trimmed_file_path = os.path.join(trimmed_file_path, file_path_list[i+1])
#     return trimmed_file_path

# str='a/b/c/d'
# str2 = trim_path_dir(str)

# print(str2)