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
ti.init(arch=ti.cuda, device_memory_GB=3) # Use GPU
# ti.init(arch=ti.cpu) # Use CPU

''' GLOBAL SETTINGS '''
part_size = 0.01
time_step = part_size/150
world = World(dim=2)
world.set_part_size(part_size)
world.set_time_step(time_step)

'''BASIC SETTINGS FOR FLUID'''
fluid_rest_density = val_f(1000)
cube_data = Cube_data(lb=vec2f(-2, -3.8), rt=vec2f(-0.2, 0), span=world.g_part_size[None]*1.001)
'''INIT AN FLUID PARTICLE OBJECT'''
fluid_part_num = val_i(cube_data.num)
fluid_part_1 = world.add_part_obj(part_num=fluid_part_num[None], is_dynamic=True)
fluid_part_1.instantiate_from_template(part_template)
'''PUSH PARTICLES TO THE OBJECT'''
fluid_part_1.open_stack(val_i(cube_data.num))
fluid_part_1.fill_open_stack_with_nparr(fluid_part_1.pos, cube_data.pos)
fluid_part_1.fill_open_stack_with_val(fluid_part_1.size, fluid_part_1.get_part_size())
fluid_part_1.fill_open_stack_with_val(fluid_part_1.volume, val_f(fluid_part_1.get_part_size()[None]**world.g_dim[None]))
fluid_part_1.fill_open_stack_with_val(fluid_part_1.mass, val_f(fluid_rest_density[None]*fluid_part_1.get_part_size()[None]**world.g_dim[None]))
fluid_part_1.fill_open_stack_with_val(fluid_part_1.rest_density, fluid_rest_density)
fluid_part_1.close_stack()

# fluid_part_cube_gen = Cube_generator(fluid_part_1, lb=vec2f(-2, -3.8), rt=vec2f(0, 0))
# print('prepared to push fluid part', fluid_part_cube_gen.pushed_num_preview(factor=1.001))
# part_num = fluid_part_cube_gen.push_pos(factor=1.001)
# fluid_part_1.set_val(to_arr=fluid_part_1.size, num=part_num, val=world.g_part_size)
# fluid_part_1.set_val(to_arr=fluid_part_1.volume, num=part_num, val=world.part_volume)
# fluid_part_1.set_val(to_arr=fluid_part_1.mass, num=part_num, val=val_f(fluid_rest_density[None]*world.part_volume[None]))
# fluid_part_1.set_val(to_arr=fluid_part_1.rest_density, num=part_num, val=fluid_rest_density)
# fluid_part_1.update_stack_top(part_num)
# print('pushed fluid part num', part_num)

'''INIT AN FLUID PARTICLE OBJECT'''
cube_data.translate(vec2f(2.2, 0))
fluid_part_num = val_i(cube_data.num)
fluid_part_2 = world.add_part_obj(part_num=fluid_part_num[None], is_dynamic=True)
fluid_part_2.instantiate_from_template(part_template)
'''PUSH PARTICLES TO THE OBJECT'''
fluid_part_2.open_stack(val_i(cube_data.num))
fluid_part_2.fill_open_stack_with_nparr(fluid_part_2.pos, cube_data.pos)
fluid_part_2.fill_open_stack_with_val(fluid_part_2.size, fluid_part_2.get_part_size())
fluid_part_2.fill_open_stack_with_val(fluid_part_2.volume, val_f(fluid_part_2.get_part_size()[None]**world.g_dim[None]))
fluid_part_2.fill_open_stack_with_val(fluid_part_2.mass, val_f(fluid_rest_density[None]*fluid_part_2.get_part_size()[None]**world.g_dim[None]))
fluid_part_2.fill_open_stack_with_val(fluid_part_2.rest_density, fluid_rest_density)
fluid_part_2.close_stack()

''' INIT BOUNDARY PARTICLE OBJECT '''
bound_part_num = val_i(5e4)
bound_rest_density = val_f(1000)
bound_part = world.add_part_obj(part_num=bound_part_num[None], is_dynamic=False)
bound_part.instantiate_from_template(part_template)
bound_box_gen = Box_generator(obj=bound_part, lb=vec2f(-4, -4), rt=vec2f(4, 4), layers=2)
print('prepared to push bound part', bound_box_gen.pushed_num_preview(factor=1.001))
part_num = bound_box_gen.push_pos(factor=1.001)
print('pushed bound part num', part_num)
bound_part.set_val(to_arr=bound_part.size, num=part_num, val=world.g_part_size)
bound_part.set_val(to_arr=bound_part.volume, num=part_num, val=world.part_volume)
bound_part.set_val(to_arr=bound_part.mass, num=part_num, val=val_f(bound_rest_density[None]*world.part_volume[None]))
bound_part.set_val(to_arr=bound_part.rest_density, num=part_num, val=bound_rest_density)
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

fluid1_neighb_search.update_self()
fluid2_neighb_search.update_self()
bound_neighb_search.update_self()

'''INIT SOLVERS'''
fluid1_adv = Adv_slover(fluid_part_1)
fluid1_df = DF_solver(fluid_part_1)
fluid2_adv = Adv_slover(fluid_part_2)
fluid2_df = DF_solver(fluid_part_2)
bound_df = DF_solver(bound_part)
df_layer = DF_layer([fluid1_df, fluid2_df, bound_df])

# sense_grid = Sense_grid(type=Sense_grid.FIXED_GRID, neighb_pool_size=val_i(3e6),world=world, cell_size=val_f(0.1))
sense_grid = Sense_grid(type=Sense_grid.FIXED_RES, neighb_pool_size=val_i(3e6), world=world, cell_size=val_f(0.1), grid_res=val_i(64), grid_center=vec2_f([0, 0]))
pos_np = sense_grid.node_index.to_numpy().astype(np.int32)
# save pos_np to txt file
np.savetxt("pos_np.txt", pos_np, fmt='%i')

# sense_grid.add_sensed_particles(fluid_part_1)
# sense_grid.add_sensed_particles(fluid_part_2)
sense_grid.step()
sense_output = Output_manager(format_type = Output_manager.type.GRID, data_source = sense_grid)
sense_output.add_output_dataType("pos",2)
sense_output.export_to_numpy()
print('DEBUG sense_output', sense_output.np_node_index_organized)
# save as numpy file
np.save("pos_np.npy", sense_output.np_node_index_organized)

def loop():
    fluid1_neighb_search.update_self()
    fluid2_neighb_search.update_self()
    bound_neighb_search.update_self()

    fluid1_neighb_search.search_neighbors()
    fluid2_neighb_search.search_neighbors()
    bound_neighb_search.search_neighbors()

    sense_grid.step()

    fluid1_adv.adv_step(in_vel= fluid_part_1.vel, out_vel_adv=fluid_part_1.vel_adv)
    fluid2_adv.adv_step(in_vel= fluid_part_2.vel, out_vel_adv=fluid_part_2.vel_adv)

    df_layer.step()

    # fluid1_adv.adv_step(in_vel= fluid_part_1.vel, out_vel_adv=fluid_part_1.vel)
    # fluid2_adv.adv_step(in_vel= fluid_part_2.vel, out_vel_adv=fluid_part_2.vel)

    fluid1_adv.update_pos(in_vel= fluid_part_1.vel, out_pos=fluid_part_1.pos)
    fluid2_adv.update_pos(in_vel= fluid_part_2.vel, out_pos=fluid_part_2.pos)


def run(loop):
    gui = Gui3d()

    fps = 60
    inv_fps = 1/fps
    timer = 0
    sim_time = 0
    loop_count = 0

    while gui.window.running:
        gui.monitor_listen()

        if gui.op_system_run:
            loop()
            loop_count += 1
            sim_time += world.g_dt[None]
            print('loop count', loop_count, 'compressible ratio', 'incompressible iter', fluid1_df.incompressible_iter[None])
        
        if gui.op_refresh_window:
            gui.scene_setup()
            if gui.show_bound:
                gui.scene_add_parts(obj_pos=fluid_part_1.pos, obj_color=(1,0.5,0),index_count=fluid_part_1.get_stack_top()[None],size=world.g_part_size[None])
                gui.scene_add_parts(obj_pos=fluid_part_2.pos, obj_color=(0,0.5,1),index_count=fluid_part_2.get_stack_top()[None],size=world.g_part_size[None])
                gui.scene_add_parts(obj_pos=bound_part.pos, obj_color=(0,0.5,1),index_count=bound_part.get_stack_top()[None],size=world.g_part_size[None])
            else:
                gui.scene_add_parts_colorful(obj_pos=sense_grid.pos, obj_color=sense_grid.clampped_rgb, index_count=sense_grid.get_stack_top()[None], size=sense_grid.get_part_size()[None]*0.5)
            
            gui.canvas.scene(gui.scene)  # Render the scene

            if(sim_time > timer*inv_fps):
                if gui.op_save_img:
                    gui.window.save_image('output/'+str(timer)+'.png')
                timer += 1

            gui.window.show()
    
        if timer > 660:
            break

loop()
run(loop)







