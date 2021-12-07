import os
import taichi as ti

ti.init()

a = ti.Vector.field(3, ti.f32, (10,4))

print(a.shape)
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