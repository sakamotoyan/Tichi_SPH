import taichi as ti
from ti_sph import *
from part_template import part_template
import time
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

''' TAICHI SETTINGS '''
# ti.init(arch=ti.cuda, kernel_profiler=True) 
ti.init(arch=ti.cuda, device_memory_GB=3) # Use GPU
# ti.init(arch=ti.cpu) # Use CPU

''' GLOBAL SETTINGS '''
world = World()

'''BASIC SETTINGS FOR FLUID'''
fluid_part_num = val_i(2e5)
fluid_rest_density = val_f(1000)
fluid_part_neighb_num = val_i(fluid_part_num[None]*world.avg_neighb_part_num[None])
'''INIT AN FLUID PARTICLE OBJECT'''
fluid_part = part_template(part_num=2e5, world=world)
'''PUSH PARTICLES TO THE OBJECT'''
fluid_part_cube_gen = Cube_generator(fluid_part, lb=vec3f(-2, -2.6, -2), rt=vec3f(1, 1, 1))
part_num = fluid_part_cube_gen.push_pos()
fluid_part.set_from_val(to_arr=fluid_part.size, num=part_num, val=world.part_size)
fluid_part.set_from_val(to_arr=fluid_part.volume, num=part_num, val=world.part_volume)
fluid_part.set_from_val(to_arr=fluid_part.mass, num=part_num, val=val_f(fluid_rest_density[None]*world.part_volume[None]))
fluid_part.set_from_val(to_arr=fluid_part.rest_density, num=part_num, val=fluid_rest_density)
fluid_part.update_stack_top(part_num)
print('pushed fluid part num', part_num)

'''INIT NEIGHBOR SEARCH OBJECTS'''
fluid_neighb_search = Neighb_search(fluid_part)
'''FLUID SOLVERS'''
fluid_adv = Adv_funcs(fluid_part)
fluid_df = DF_solver(fluid_part)


''' INIT BOUNDARY PARTICLE OBJECT '''
bound_part_num = val_i(2e5)
bound_rest_density = val_f(1000)
bound_part_neighb_num = val_i(bound_part_num[None]*world.avg_neighb_part_num[None]) #vali_()
bound_part = part_template(part_num=bound_part_num[None], world=world)
bound_part.is_dynamic = False
bound_box_gen = Box_generator(obj=bound_part, lb=vec3f(-3, -3, -3), rt=vec3f(3,3,3), layers=2)
part_num = bound_box_gen.push_pos()
print('pushed bound part num', part_num)
bound_part.set_from_val(to_arr=bound_part.size, num=part_num, val=world.part_size)
bound_part.set_from_val(to_arr=bound_part.volume, num=part_num, val=world.part_volume)
bound_part.set_from_val(to_arr=bound_part.mass, num=part_num, val=val_f(bound_rest_density[None]*world.part_volume[None]))
bound_part.set_from_val(to_arr=bound_part.rest_density, num=part_num, val=bound_rest_density)
bound_part.update_stack_top(part_num)
bound_neighb_search = Neighb_search(bound_part)
bound_df = DF_solver(bound_part)

fluid_neighb_search.add_neighb(fluid_part, world.support_radius)
fluid_neighb_search.add_neighb(bound_part, world.support_radius)

bound_neighb_search.add_neighb(bound_part, world.support_radius)
bound_neighb_search.add_neighb(fluid_part, world.support_radius)

fluid_neighb_search.update_self()
bound_neighb_search.update_self()

fluid_neighb_search.search_neighbors()
bound_neighb_search.search_neighbors()



def loop():

    fluid_adv.adv_step(in_vel= fluid_part.vel, out_vel_adv=fluid_part.vel_adv)
    fluid_df.df_step_dynamic_phase(in_vel_adv = fluid_part.vel_adv, out_vel_ = fluid_part.vel, neighb_list__=fluid_neighb)
    fluid_adv.update_pos(in_vel= fluid_part.vel, out_pos=fluid_part.pos_)

def loop_tmp():
    fluid_neighb_search.update_self()
    bound_neighb_search.update_self()

    fluid_neighb_search.search_neighbors()
    bound_neighb_search.search_neighbors()

    fluid_df.df_step_static_phase(neighb_list=fluid_neighb_search.neighb_list)
    bound_df.df_step_static_phase(neighb_list=bound_neighb_search.neighb_list)

    fluid_adv.adv_step(in_vel= fluid_part.vel, out_vel_adv=fluid_part.vel_adv)

    fluid_df.get_vel_adv(in_vel_adv=fluid_part.vel_adv)
    while True:
        fluid_df.incompressible_iter_[None] += 1

        ''' Compute Delta Density '''
        fluid_df.compute_a_delta_density()
        bound_df.compute_a_delta_density()

        ''' Further Update Delta Density '''
        for neighb_obj__ in fluid_neighb_search.neighb_list.neighb_obj_list:
            fluid_df.loop_neighb(fluid_neighb_search.neighb_list, neighb_obj__, fluid_df.inloop_compute_u_delta_density_from_vel_adv)
        for neighb_obj__ in bound_neighb_search.neighb_list.neighb_obj_list:
            bound_df.loop_neighb(bound_neighb_search.neighb_list, neighb_obj__, bound_df.inloop_compute_u_delta_density_from_vel_adv)
        
        fluid_df.ReLU_a_delta_density()
        bound_df.ReLU_a_delta_density()

        ''' Update Density Ratio from Delta Density '''
        fluid_df.update_compressible_ratio()
        bound_df.update_compressible_ratio()
        ''' Incompressible Condition '''
        if not (fluid_df.compressible_ratio_[None] > fluid_df.incompressible_threshold_[None] \
                and fluid_df.incompressible_iter_[None] < fluid_df.incompressible_iter_max_[None]\
                    # and bound_df.compressible_ratio_[None] > bound_df.incompressible_threshold_[None]\
                        ):
            break
        
        for neighb_obj__ in fluid_neighb_search.neighb_list.neighb_obj_list:
            ''' Further Update Delta Density '''
            fluid_df.loop_neighb(fluid_neighb_search.neighb_list, neighb_obj__, fluid_df.inloop_compute_u_vel_adv_from_alpha)

    fluid_df.update_vel(fluid_part.vel)

    fluid_adv.update_pos(in_vel= fluid_part.vel, out_pos=fluid_part.pos)

gui = Gui3d()
loop_count = 0
loop_tmp()
while gui.window.running:
    if gui.op_system_run:
        loop_tmp()
        loop_count += 1

    gui.monitor_listen()

    if gui.op_refresh_window:
        gui.scene_setup()
        gui.scene_add_parts(obj_pos=fluid_part.pos, obj_color=(1,0.5,0),index_count=fluid_part.get_stack_top()[None],size=world.part_size[None])
        if gui.show_bound:
            gui.scene_add_parts(obj_pos=bound_part.pos, obj_color=(0,0.5,1),index_count=bound_part.get_stack_top()[None],size=world.part_size[None])
        gui.scene_render()





