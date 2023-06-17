import taichi as ti
from ti_sph import *
from part_template import part_template
from grid_template import grid_template
import time
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

''' TAICHI SETTINGS '''
# ti.init(arch=ti.cuda, kernel_profiler=True) 
ti.init(arch=ti.cuda, device_memory_GB=4) # Use GPU
# ti.init(arch=ti.cpu) # Use CPU

''' GLOBAL SETTINGS '''
world = World(dim=2)
world.set_part_size(0.02)
world.set_dt(world.g_part_size[None]/world.g_sound_speed[None])

'''BASIC SETTINGS FOR FLUID'''
fluid_part_num = val_i(5e4)
fluid_rest_density = val_f(1000)
'''INIT AN FLUID PARTICLE OBJECT'''
fluid_part_1 = part_template(part_num=fluid_part_num[None], world=world)
'''PUSH PARTICLES TO THE OBJECT'''
fluid_part_cube_gen = Cube_generator(fluid_part_1, lb=vec2f(-2, -3), rt=vec2f(0, 1))
print('prepared to push fluid part', fluid_part_cube_gen.pushed_num_preview(factor=1.001))
part_num = fluid_part_cube_gen.push_pos(factor=1.001)
fluid_part_1.set_from_val(to_arr=fluid_part_1.size, num=part_num, val=world.g_part_size)
fluid_part_1.set_from_val(to_arr=fluid_part_1.volume, num=part_num, val=world.part_volume)
fluid_part_1.set_from_val(to_arr=fluid_part_1.mass, num=part_num, val=val_f(fluid_rest_density[None]*world.part_volume[None]))
fluid_part_1.set_from_val(to_arr=fluid_part_1.rest_density, num=part_num, val=fluid_rest_density)
fluid_part_1.update_stack_top(part_num)
print('pushed fluid part num', part_num)

'''INIT AN FLUID PARTICLE OBJECT'''
fluid_part_2 = part_template(part_num=fluid_part_num[None], world=world)
'''PUSH PARTICLES TO THE OBJECT'''
fluid_part_cube_gen = Cube_generator(fluid_part_2, lb=vec2f(1, -3), rt=vec2f(3, 1))
part_num = fluid_part_cube_gen.push_pos(factor=1.001)
fluid_part_2.set_from_val(to_arr=fluid_part_2.size, num=part_num, val=world.g_part_size)
fluid_part_2.set_from_val(to_arr=fluid_part_2.volume, num=part_num, val=world.part_volume)
fluid_part_2.set_from_val(to_arr=fluid_part_2.mass, num=part_num, val=val_f(fluid_rest_density[None]*world.part_volume[None]))
fluid_part_2.set_from_val(to_arr=fluid_part_2.rest_density, num=part_num, val=fluid_rest_density)
fluid_part_2.update_stack_top(part_num)
print('pushed fluid part num', part_num)

''' INIT BOUNDARY PARTICLE OBJECT '''
bound_part_num = val_i(5e4)
bound_rest_density = val_f(1000)
bound_part = part_template(part_num=bound_part_num[None], world=world)
bound_part.is_dynamic = False
bound_box_gen = Box_generator(obj=bound_part, lb=vec2f(-4, -4), rt=vec2f(4, 4), layers=2)
print('prepared to push bound part', bound_box_gen.pushed_num_preview(factor=1.001))
part_num = bound_box_gen.push_pos(factor=1.001)
print('pushed bound part num', part_num)
bound_part.set_from_val(to_arr=bound_part.size, num=part_num, val=world.g_part_size)
bound_part.set_from_val(to_arr=bound_part.volume, num=part_num, val=world.part_volume)
bound_part.set_from_val(to_arr=bound_part.mass, num=part_num, val=val_f(bound_rest_density[None]*world.part_volume[None]))
bound_part.set_from_val(to_arr=bound_part.rest_density, num=part_num, val=bound_rest_density)
bound_part.update_stack_top(part_num)

'''INIT NEIGHBOR SEARCH OBJECTS'''
fluid1_neighb_search = Neighb_search(fluid_part_1)
fluid2_neighb_search = Neighb_search(fluid_part_2)
bound_neighb_search = Neighb_search(bound_part)

fluid1_neighb_search.add_neighb_obj(fluid_part_1, world.support_radius)
fluid1_neighb_search.add_neighb_obj(fluid_part_2, world.support_radius)
fluid1_neighb_search.add_neighb_obj(bound_part, world.support_radius)
fluid2_neighb_search.add_neighb_obj(fluid_part_1, world.support_radius)
fluid2_neighb_search.add_neighb_obj(fluid_part_2, world.support_radius)
fluid2_neighb_search.add_neighb_obj(bound_part, world.support_radius)
bound_neighb_search.add_neighb_obj(bound_part, world.support_radius)
bound_neighb_search.add_neighb_obj(fluid_part_1, world.support_radius)
bound_neighb_search.add_neighb_obj(fluid_part_2, world.support_radius)

'''INIT SOLVERS'''
fluid1_adv = Adv_slover(fluid_part_1)
fluid1_df = DF_solver(fluid_part_1)
fluid2_adv = Adv_slover(fluid_part_2)
fluid2_df = DF_solver(fluid_part_2)
bound_df = DF_solver(bound_part)
df_layer = DF_layer([fluid1_df, fluid2_df, bound_df])

def loop():
    fluid1_neighb_search.update_self()
    fluid2_neighb_search.update_self()
    bound_neighb_search.update_self()

    fluid1_neighb_search.search_neighbors()
    fluid2_neighb_search.search_neighbors()
    bound_neighb_search.search_neighbors()

    fluid1_adv.adv_step(in_vel= fluid_part_1.vel, out_vel_adv=fluid_part_1.vel_adv)
    fluid2_adv.adv_step(in_vel= fluid_part_2.vel, out_vel_adv=fluid_part_2.vel_adv)

    df_layer.step()

    # fluid1_adv.adv_step(in_vel= fluid_part_1.vel, out_vel_adv=fluid_part_1.vel)
    # fluid2_adv.adv_step(in_vel= fluid_part_2.vel, out_vel_adv=fluid_part_2.vel)

    fluid1_adv.update_pos(in_vel= fluid_part_1.vel, out_pos=fluid_part_1.pos)
    fluid2_adv.update_pos(in_vel= fluid_part_2.vel, out_pos=fluid_part_2.pos)

    print('loop count', loop_count, 'compressible ratio', 'incompressible iter', fluid1_df.incompressible_iter[None])



loop_count = 0
loop()
print(fluid_part_1.sph.density.to_numpy()[0:100])

gui = Gui3d()
while gui.window.running:
    if gui.op_system_run:
        loop()
        loop_count += 1

    gui.monitor_listen()

    if gui.op_refresh_window:
        gui.scene_setup()
        gui.scene_add_parts(obj_pos=fluid_part_1.pos, obj_color=(1,0.5,0),index_count=fluid_part_1.get_stack_top()[None],size=world.g_part_size[None])
        gui.scene_add_parts(obj_pos=fluid_part_2.pos, obj_color=(0,0.5,1),index_count=fluid_part_2.get_stack_top()[None],size=world.g_part_size[None])
        if gui.show_bound:
            gui.scene_add_parts(obj_pos=bound_part.pos, obj_color=(0,0.5,1),index_count=bound_part.get_stack_top()[None],size=world.g_part_size[None])
        gui.scene_render()





