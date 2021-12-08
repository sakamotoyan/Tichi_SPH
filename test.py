import os

# import taichi as ti
# ti.init()

# a = ti.Vector.field(3, ti.f32, (10,4))

def func(p1,p2):
    print(p1+p2)

def func_factory(func, *func_params):
    func(*func_params)

func_params = (1,2)
func_factory(func, *func_params)

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