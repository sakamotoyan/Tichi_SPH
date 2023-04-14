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
g_dim = val_i(3)
g_simspace_lb = vec3_f([-8, -8, -8])  # simulation space lower bound
g_simspace_rt = vec3_f([8, 8, 8])  # simulation space upper bound
g_part_size = val_f(0.07)  # particle size
g_part_volume = val_f(pow(g_part_size[None], g_dim[None]))
g_supporrt_radius = val_f(2*g_part_size[None])  # support radius
g_gravity = vec3_f([0, -9.8, 0])  # gravity
g_dt = val_f(2e-3)  # time step
g_rest_density = val_f(1000)  # rest density
g_kinematic_viscosity = val_f(1e-3)  # kinematic viscosity
g_max_obj_nums = val_i(5)


'''BASIC SETTINGS FOR FLUID'''
fluid_part_num = val_i(2e5)
fluid_part_neighb_num = val_i(fluid_part_num[None]*30)
'''INIT AN FLUID PARTICLE OBJECT'''
fluid_part = part_template(part_num=fluid_part_num[None], dim=g_dim[None], verbose=False)
fluid_part.id = 0
fluid_part.is_dynamic = True
'''PUSH PARTICLES TO THE OBJECT'''
fluid_part_cube_gen = Cube_generator(lb=vec3f(-2, -2.6, -2), rt=vec3f(1, 1, 1))
part_num = fluid_part_cube_gen.push_pos_based_on_span(span=g_part_size[None], obj_pos_=fluid_part.pos_, obj_stack_top_=fluid_part.stack_top_)
print('pushed fluid part num', part_num)
fluid_part.push_from_val_(to_=fluid_part.size_, num=part_num, val_=g_part_size)
fluid_part.push_from_val_(to_=fluid_part.volume_, num=part_num, val_=g_part_volume)
fluid_part.push_from_val_(to_=fluid_part.mass_, num=part_num, val_=val_f(g_rest_density[None]*g_part_volume[None]))
fluid_part.push_from_val_(to_=fluid_part.rest_density_, num=part_num, val_=g_rest_density)
fluid_part.update_stack_top(part_num)
'''INIT NEIGHBOR SEARCH OBJECTS'''
fluid_neighb_cell = Neighb_cell_simple(obj__=fluid_part, cell_size_=g_supporrt_radius, lb_=g_simspace_lb, 
rt_=g_simspace_rt, part_num_=fluid_part.part_num_, stack_top_=fluid_part.stack_top_, pos_=fluid_part.pos_)
fluid_neighb_cell.update_part_in_cell()
'''FLUID SOLVERS'''
fluid_adv = Adv_funcs(fluid_part, g_dt, g_gravity, g_kinematic_viscosity)
fluid_df = DF_solver(fluid_part, g_dt)


''' INIT BOUNDARY PARTICLE OBJECT '''
bound_part_num = val_i(2e5)
bound_part_neighb_num = val_i(bound_part_num[None]*30) #vali_()
bound_part = part_template(part_num=bound_part_num[None], dim=g_dim[None], verbose=False)
bound_part.id = 1
bound_part.is_dynamic = False
bound_box_gen = Box_generator(lb=vec3f(-3, -3, -3), rt=vec3f(3,3,3), layers=2)
part_num = bound_box_gen.push_pos_based_on_span(span=g_part_size[None], obj_pos_=bound_part.pos_, obj_stack_top_=bound_part.stack_top_)
print('pushed bound part num', part_num)
bound_part.push_from_val_(to_=bound_part.size_, num=part_num, val_=g_part_size)
bound_part.push_from_val_(to_=bound_part.volume_, num=part_num, val_=g_part_volume)
bound_part.push_from_val_(to_=bound_part.mass_, num=part_num, val_=val_f(g_rest_density[None]*g_part_volume[None]))
bound_part.push_from_val_(to_=bound_part.rest_density_, num=part_num, val_=g_rest_density)
bound_part.update_stack_top(part_num)
bound_neighb_cell = Neighb_cell_simple(obj__=bound_part, cell_size_=g_supporrt_radius, lb_=g_simspace_lb, 
    rt_=g_simspace_rt, part_num_=bound_part.part_num_, stack_top_=bound_part.stack_top_, pos_=bound_part.pos_)
bound_neighb_cell.update_part_in_cell()
bound_df = DF_solver(bound_part, g_dt)


fluid_neighb = Neighb_list(obj__=fluid_part, obj_pos_=fluid_part.pos_, obj_stack_top_=fluid_part.stack_top_, max_neighb_part_num_=fluid_part_neighb_num, max_neighb_obj_num_=g_max_obj_nums)
fluid_neighb.add_neighb_obj(neighb_obj__=fluid_part, neighb_obj_pos_=fluid_part.pos_, neighb_cell__=fluid_neighb_cell, search_range_=g_supporrt_radius)
fluid_neighb.add_neighb_obj(neighb_obj__=bound_part, neighb_obj_pos_=bound_part.pos_, neighb_cell__=bound_neighb_cell, search_range_=g_supporrt_radius)
fluid_neighb.register_neighbours()

bound_neighb = Neighb_list(obj__=bound_part, obj_pos_=bound_part.pos_, obj_stack_top_=bound_part.stack_top_, max_neighb_part_num_=bound_part_neighb_num, max_neighb_obj_num_=g_max_obj_nums)
bound_neighb.add_neighb_obj(neighb_obj__=fluid_part, neighb_obj_pos_=fluid_part.pos_, neighb_cell__=fluid_neighb_cell, search_range_=g_supporrt_radius)
bound_neighb.add_neighb_obj(neighb_obj__=bound_part, neighb_obj_pos_=bound_part.pos_, neighb_cell__=bound_neighb_cell, search_range_=g_supporrt_radius)
bound_neighb.register_neighbours()


def loop():
    fluid_neighb_cell.update_part_in_cell()
    bound_neighb_cell.update_part_in_cell()
    fluid_neighb.register_neighbours()
    bound_neighb.register_neighbours()

    fluid_df.df_step_static_phase(neighb_list__=fluid_neighb)
    bound_df.df_step_static_phase(neighb_list__=bound_neighb)

    fluid_adv.adv_step(in_vel_= fluid_part.vel_, out_vel_adv_=fluid_part.vel_adv_)
    fluid_df.df_step_dynamic_phase(in_vel_adv_ = fluid_part.vel_adv_, out_vel_ = fluid_part.vel_, neighb_list__=fluid_neighb)
    fluid_adv.update_pos(in_vel_= fluid_part.vel_, out_pos_=fluid_part.pos_)

def loop_tmp():
    fluid_neighb_cell.update_part_in_cell()
    bound_neighb_cell.update_part_in_cell()
    fluid_neighb.register_neighbours()
    bound_neighb.register_neighbours()

    fluid_df.df_step_static_phase(neighb_list__=fluid_neighb)
    bound_df.df_step_static_phase(neighb_list__=bound_neighb)

    fluid_adv.adv_step(in_vel_= fluid_part.vel_, out_vel_adv_=fluid_part.vel_adv_)

    fluid_df.get_vel_adv(in_vel_adv_=fluid_part.vel_adv_)
    while True:
        fluid_df.incompressible_iter_[None] += 1

        ''' Compute Delta Density '''
        fluid_df.compute_a_delta_density()
        bound_df.compute_a_delta_density()

        ''' Further Update Delta Density '''
        for neighb_obj__ in fluid_neighb.neighb_obj_list:
            fluid_df.loop_neighb(fluid_neighb, neighb_obj__, fluid_df.inloop_compute_u_delta_density_from_vel_adv)
        for neighb_obj__ in bound_neighb.neighb_obj_list:
            bound_df.loop_neighb(bound_neighb, neighb_obj__, bound_df.inloop_compute_u_delta_density_from_vel_adv)
        
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
        
        for neighb_obj__ in fluid_neighb.neighb_obj_list:
            ''' Further Update Delta Density '''
            fluid_df.loop_neighb(fluid_neighb, neighb_obj__, fluid_df.inloop_compute_u_vel_adv_from_alpha)

    fluid_df.update_vel(fluid_part.vel_)

    fluid_adv.update_pos(in_vel_= fluid_part.vel_, out_pos_=fluid_part.pos_)

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
        gui.scene_add_parts(obj_pos=fluid_part.pos_, obj_color=(1,0.5,0),index_count=fluid_part.stack_top_[None],size=g_part_size[None])
        if gui.show_bound:
            gui.scene_add_parts(obj_pos=bound_part.pos_, obj_color=(0,0.5,1),index_count=bound_part.stack_top_[None],size=g_part_size[None])
        gui.scene_render()





