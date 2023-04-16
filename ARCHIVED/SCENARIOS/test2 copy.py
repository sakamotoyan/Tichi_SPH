import taichi as ti
from ti_sph import *
from SIM_SCENE.part_template import fluid_part_template
import time
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

## TAICHI SETTINGS
# ti.init(arch=ti.cuda, kernel_profiler=True) 
ti.init(arch=ti.cuda, device_memory_GB=3) # Use GPU
# ti.init(arch=ti.cpu) # Use CPU

## GLOBAL SETTINGS
g_part_size = val_f(1)  # particle size
g_supporrt_radius = val_f(2*g_part_size[None])  # support radius
g_gravity = vec3_f([0, -9.8, 0])  # gravity
g_dt = val_f(1e-3)  # time step
g_rest_density = val_f(1000)  # rest density
g_kinematic_viscosity = val_f(1e-3)  # kinematic viscosity
g_simspace_lb = vec3_f([-8, -8, -8])  # simulation space lower bound
g_simspace_rt = vec3_f([8, 8, 8])  # simulation space upper bound
g_dim = val_i(3)

## BASIC SETTINGS FOR FLUID
fluid_part_num = val_i(1e5)
## INIT AN FLUID PARTICLE OBJECT
fluid_part = fluid_part_template(part_num=fluid_part_num[None], dim=g_dim[None], verbose=False)

## GENERATE FLUID PARTICLE POSITIONS AND PUSH THEM TO THE PARTICLE OBJECT
fluid_part_cube_gen = Cube_generator(lb=vec3f(-5, -2, -7), rt=vec3f(2, 4, -2))
fluid_part_cube_gen.push_pos_based_on_span(span=g_part_size[None], pos=fluid_part.pos, stack_top=fluid_part.stack_top)
# fluid_part_cube_gen.generate_pos_based_on_span(span=g_part_size[None])
# fluid_part.set_from_numpy(fluid_part.pos, fluid_part_cube_gen.np_pos)
fluid_part.update_stack_top(fluid_part_cube_gen.num)
print(fluid_part.pos.to_numpy()[0:1])

## INIT NEIGHBOR SEARCH OBJECTS
fluid_part_NS_f = Neighb_search_FS(dim=g_dim, cell_size=g_supporrt_radius, lb=g_simspace_lb,
                            rt=g_simspace_rt, obj=fluid_part, obj_pos=fluid_part.pos)

fluid_part_NS_old = Neighb_grid_slim(
    part_num=fluid_part.part_num[None],
    dim=3,
    lb=g_simspace_lb[None],
    rt=g_simspace_rt[None],
    cell_size=g_supporrt_radius[None],
)

fluid_part_NS_s = Neighb_search_s(dim=g_dim, cell_size=g_supporrt_radius, lb=g_simspace_lb, 
    rt=g_simspace_rt, part_num=fluid_part.part_num, stack_top=fluid_part.stack_top, pos=fluid_part.pos)

delta_x = 1/500
iter = 8000
move_pos = vec3_f([0, 0, delta_x])

# pre_heat
for i in range(1):
    fluid_part_NS_f.update_part_in_cell()
    fluid_part_NS_s.update_part_in_cell()

    # fluid_part_NS_f.get_part_in_cell()
    fluid_part_NS_s.get_part_in_cell()

    fluid_part_NS_old.register(fluid_part.stack_top, fluid_part.pos)





# start_time = time.time()
# for i in range(iter):
    # ker_arr_add(fluid_part.pos, move_pos, 0, fluid_part.stack_top[None])
    # fluid_part_NS_f.update_part_in_cell()
    # fluid_part_NS_f.get_part_in_cell()

    # fluid_part_NS_s.update_part_in_cell()
    # fluid_part_NS_s.get_part_in_cell()
    
    # fluid_part_NS_old.register(fluid_part.stack_top, fluid_part.pos)

    # fluid_part_NS.get_part_in_cell()
    # fluid_part_NS.time_test()
# end_time = time.time()
# print("time cost: ", end_time-start_time)



print("GOING...")


inter_start_time = 0
inter_end_time = 0

acc_time = 0
inter_start_time = time.time()
for i in range(iter):
    # ker_arr_add(fluid_part.pos, move_pos, 0, fluid_part.stack_top[None])
    fluid_part_NS_s.update_part_in_cell()
inter_end_time = time.time()
acc_time += inter_end_time - inter_start_time
print("time cost: ", acc_time)

acc_time = 0
inter_start_time = time.time()
for i in range(iter):
    # ker_arr_add(fluid_part.pos, move_pos, 0, fluid_part.stack_top[None])
    fluid_part_NS_old.register(fluid_part.stack_top, fluid_part.pos)
    inter_end_time = time.time()
acc_time += inter_end_time - inter_start_time
end_time = time.time()
print("time cost: ", acc_time)

# fluid_part_NS_f.get_part_in_cell()
# fluid_part_NS_s.get_part_in_cell()

print("FINISHING...")


