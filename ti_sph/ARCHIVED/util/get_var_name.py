import taichi as ti
import inspect


def get_variable_name(variable):
    # print(locals())
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()

    for var_name, var_val in callers_local_vars:
        if var_val is variable:
            return var_name
    # return [f'{var_name}: {var_val}' for var_name, var_val in callers_local_vars if var_val is variable]

""" below is the test code: """  

# a = 2

# struct_node_basic = ti.types.struct(
#         pos=ti.types.vector(3, ti.f32),   # position
#         vel=ti.types.vector(3, ti.f32),   # velocity
#         acc=ti.types.vector(3, ti.f32),   # acceleration (not always used)
#         force=ti.types.vector(3, ti.f32), # force (not always used)
#         mass=ti.f32,                        # mass
#         rest_density=ti.f32,                # rest density
#         rest_volume=ti.f32,                 # rest volume
#         size=ti.f32,                        # diameter
#     )

# struct_node_basic2 = ti.types.struct(
#         pos=ti.types.vector(3, ti.f32),   # position
#         vel=ti.types.vector(3, ti.f32),   # velocity
#         acc=ti.types.vector(3, ti.f32),   # acceleration (not always used)
#         force=ti.types.vector(3, ti.f32), # force (not always used)
#         mass=ti.f32,                        # mass
#         rest_density=ti.f32,                # rest density
#         rest_volume=ti.f32,                 # rest volume
#         size=ti.f32,                        # diameter
#     )

# struct_list = [struct_node_basic, struct_node_basic2]

# for struct in struct_list:
#     print(get_variable_name(struct))