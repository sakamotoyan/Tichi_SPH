import numpy
from sph import *
import json
import os

'''make folders'''
folder_name=f"{'VF' if use_VF else 'DF'}"
try:
    os.mkdir(f"{folder_name}")
    os.mkdir(f"{folder_name}\\json")
    os.mkdir(f"{folder_name}\\grid_data")
    os.mkdir(f"{folder_name}\\part_data")
    os.mkdir(f"{folder_name}\\img")
except FileExistsError:
    pass


""" init data structure """
ngrid = Ngrid()
fluid = Fluid(max_part_num=fluid_part_num)
bound = Fluid(max_part_num=bound_part_num)
grid = Grid(tuple((np_grid_size/grid_dist).astype(np.int32)),np_grid_lb,grid_dist)

for obj in obj_list:
    obj.set_zero()
    obj.ones.fill(1)

dt[None] = init_part_size/cs
phase_rest_density.from_numpy(np_phase_rest_density)
sim_space_lb.from_numpy(np_sim_space_lb)
sim_space_rt.from_numpy(np_sim_space_rt)
part_size.from_numpy(np_part_size)
sph_h.from_numpy(np_sph_h)
sph_sig.from_numpy(np_sph_sig)
gravity.from_numpy(np_gravity)
node_dim.from_numpy(np_node_dim)
node_dim_coder.from_numpy(np_node_dim_coder)
fbm_diffusion_term[None] = init_fbm_diffusion_term
fbm_convection_term[None] = init_fbm_convection_term
for i in range(np_neighb_template.shape[1]):
    for j in range(dim):
        neighb_template[i][j] = np_neighb_template[j][i]

div_iter_count = 0
incom_iter_count = 0

assign_phase_color(0x6F7DBC,0)
assign_phase_color(0xeacd76,1)

""" setup scene """
try:
    scenario=json.load(open(scenario_file))
    for part in scenario:
        ob=None
        if part=='fluid':
            ob=fluid
        elif part=='bound':
            ob=bound
        if ob is not None:
            for a in scenario[part]:
                if a['type']=='cube':
                    ob.scene_add_cube(a['start_pos'],a['end_pos'],a['volume_frac'],int(a['color'],16))
                elif a['type']=='box':
                    ob.scene_add_box(a['start_pos'],a['end_pos'],a['layers'],a['volume_frac'],int(a['color'],16))
except Exception:
    print('no scenario file or scenario file invalid, use default scenario')
    bound.scene_add_box([-1.5,-1.5,-1.5],[1.5,1.5,1.5],2,[1,0],0xaaaaaa)
    """ setup 3d scene from ply"""
    bunny_verts = read_ply('ply_models/bunny_0.05.ply')
    f_part_num = len(bunny_verts)
    fluid.push_part_from_ply(f_part_num, bunny_verts, volume_frac=[0, 1], color=0x068587)

""" write scene data """
grid_data={
    'grid_count':int(grid.size),
    'grid_lb':[float(grid.lb[i]) for i in range(len(grid.lb))],
    'init_part_size':init_part_size,
    'fluid_part_count':fluid.part_num[None],
    'bound_part_count':bound.part_num[None],
    'dt':dt[None],
    'frame_rate':refreshing_rate
}
json.dump(grid_data,open(f"{folder_name}\\data.json","w"))


# cube_verts = read_ply('ply_models/cube_0.05.ply')
# b_part_num = len(cube_verts)
# bound.push_part_from_ply(b_part_num, cube_verts, volume_frac=[0, 1], color=0xFF4500)

def sph_step():
    global div_iter_count, incom_iter_count
    """ neighbour search """
    ngrid.clear_node()
    ngrid.encode(fluid)
    ngrid.encode(bound)
    ngrid.mem_shift()
    ngrid.fill_node(fluid)
    ngrid.fill_node(bound)
    """ SPH clean value """
    SPH_clean_value(fluid)
    SPH_clean_value(bound)
    """ SPH compute W and W_grad """
    SPH_prepare_attr(ngrid, fluid, fluid)
    SPH_prepare_attr(ngrid, fluid, bound)
    SPH_prepare_attr(ngrid, bound, bound)
    SPH_prepare_attr(ngrid, bound, fluid)
    SPH_prepare_alpha_1(ngrid, fluid, fluid)
    SPH_prepare_alpha_1(ngrid, fluid, bound)
    SPH_prepare_alpha_2(ngrid, fluid, fluid)
    SPH_prepare_alpha_2(ngrid, bound, fluid)
    SPH_prepare_alpha(fluid)
    SPH_prepare_alpha(bound)
    """ IPPE SPH divergence """
    div_iter_count = 0
    SPH_vel_2_vel_adv(fluid)
    while div_iter_count<iter_threshold_min or fluid.compression[None]>divergence_threshold:
        IPPE_adv_psi_init(fluid)
        # IPPE_adv_psi_init(bound)
        IPPE_adv_psi(ngrid, fluid, fluid)
        IPPE_adv_psi(ngrid, fluid, bound)
        # IPPE_adv_psi(ngrid, bound, fluid)
        IPPE_psi_adv_non_negative(fluid)
        # IPPE_psi_adv_non_negative(bound)
        IPPE_update_vel_adv(ngrid, fluid, fluid)
        IPPE_update_vel_adv(ngrid, fluid, bound)
        div_iter_count+=1
        if div_iter_count>iter_threshold_max:
            break
    SPH_vel_adv_2_vel(fluid)
    """ SPH advection """
    SPH_advection_gravity_acc(fluid)
    SPH_advection_viscosity_acc(ngrid, fluid, fluid)
    SPH_advection_surface_tension_acc(ngrid, fluid, fluid)
    SPH_advection_update_vel_adv(fluid)
    """ IPPE SPH pressure """
    incom_iter_count = 0
    while incom_iter_count<iter_threshold_min or fluid.compression[None]>compression_threshold:
        IPPE_adv_psi_init(fluid)
        # IPPE_adv_psi_init(bound)
        IPPE_adv_psi(ngrid, fluid, fluid)
        IPPE_adv_psi(ngrid, fluid, bound)
        # IPPE_adv_psi(ngrid, bound, fluid)
        IPPE_psi_adv_non_negative(fluid)
        # IPPE_psi_adv_non_negative(bound)
        IPPE_update_vel_adv(ngrid, fluid, fluid)
        IPPE_update_vel_adv(ngrid, fluid, bound)
        incom_iter_count+=1
        if incom_iter_count>iter_threshold_max:
            break
    """ debug info """
    # print('iter div: ', div_iter_count)
    # print('incom div: ', incom_iter_count)
    """ WC SPH pressure """
    # WC_pressure_val(fluid)
    # WC_pressure_acce(ngrid, fluid, fluid)
    # WC_pressure_acce(ngrid, fluid, bound)
    # SPH_advection_update_vel_adv(fluid)
    """ FBM procedure """
    # while fluid.general_flag[None] > 0:
    #     SPH_FBM_clean_tmp(fluid)
    #     SPH_FBM_convect(ngrid, fluid, fluid)
    #     SPH_FBM_diffuse(ngrid, fluid, fluid)
    #     SPH_FBM_check_tmp(fluid)
    """ SPH update """
    # SPH_update_volume_frac(fluid)
    SPH_update_mass(fluid)
    SPH_update_pos(fluid)
    SPH_update_energy(fluid)
    map_velocity(ngrid, grid,fluid)
    return div_iter_count, incom_iter_count
    """ SPH debug """

def write_json():
    data={
        "step": step_counter,
        "frame": time_counter,
        "timeInSimulation": time_count,
        "timeStep": dt[None],
        "fps":refreshing_rate,
        "iteration" : {
            "divergenceFree_iteration" : div_iter_count,
            "incompressible_iteration" : incom_iter_count,
            "sum_iteration" :div_iter_count+incom_iter_count
        },
        "energy" :{
            "kinetic_energy":fluid.kinetic_energy[None],
            "gravity_potential_energy":fluid.gravity_potential_energy[None],
            "sum_energy":fluid.kinetic_energy[None]+fluid.gravity_potential_energy[None]
        }
    }
    s = json.dumps(data)
    with open("json\\"+ ("VF" if use_VF else "DF") + str(step_counter) + ".json","w") as f:
        f.write(s)

def write_full_json(fname):
    global frame_div_iter, frame_incom_iter
    data={
        "step": step_counter,
        "frame": time_counter,
        "timeInSimulation": time_count,
        "timeStep": dt[None],
        "fps":refreshing_rate,
        "iteration" : {
            "divergenceFree_iteration" : frame_div_iter,
            "incompressible_iteration" : frame_incom_iter,
            "sum_iteration" :frame_div_iter+frame_incom_iter
        },
        "energy" :{
            "kinetic_energy":fluid.kinetic_energy[None],
            "gravity_potential_energy":fluid.gravity_potential_energy[None],
            "sum_energy":fluid.kinetic_energy[None]+fluid.gravity_potential_energy[None]
        }
        #,
        # "info":
        # [
        #     "index",
        #     "position_x",
        #     "position_y",
        #     "volumeFraction_1",
        #     "volumeFraction_2",
        # ],
        # "data":[]
    }
    # for i in range(fluid.part_num[None]):
    #     info=[]
    #     info.append(i)
    #     info.append(fluid.pos[i][0])
    #     info.append(fluid.pos[i][1])
    #     info.append(fluid.volume_frac[i][0])
    #     info.append(fluid.volume_frac[i][1])
    #     data["data"].append(info)
    s = json.dumps(data)
    with open(fname,"w") as f:
        f.write(s)

"""2d and record data"""
if dim == 2:
    """ GUI system """
    time_count = float(0)
    time_counter = int(0)
    step_counter = int(0)
    frame_div_iter=0
    frame_incom_iter=0
    flg = True
    print('fluid particle count: ', fluid.part_num[None])
    print('bound particle count: ', bound.part_num[None])
    print('grid count:',grid.size)
    numpy.save(f"{folder_name}\\grid_data\\pos",grid.pos.to_numpy())
    gui = ti.GUI('SPH', to_gui_res(gui_res_0))
    while gui.running and not gui.get_event(gui.ESCAPE) and time_counter<1000:
        print('current time: ', time_count)
        print('time step: ', dt[None])

        '''update gui'''
        gui.clear(0xffffff)
        gui.circles(to_gui_pos(fluid), radius=to_gui_radii(part_radii_relax), color=to_gui_color(fluid))
        gui.circles(to_gui_pos(bound), radius=to_gui_radii(part_radii_relax), color=to_gui_color(bound))
        grid_vel = grid.vel.to_numpy()
        gui.circles(to_gui_pos_np(grid.pos.to_numpy().reshape((grid.size,dim))), radius=to_gui_radii(part_radii_relax)*0.5, color=0x000000)
        gui.show(f"{folder_name}\\img\\rf{int(refreshing_rate+1e-5)}_{time_counter}.png")

        '''save data'''
        write_full_json(f"{folder_name}\\json\\"+ "frame"+ str(time_counter) + ".json")
        print("div iter:",frame_div_iter,",frame iter:",frame_incom_iter)
        print('sum grid vel:',np.sum(grid.vel.to_numpy()))
        numpy.save(f"{folder_name}\\grid_data\\vel_{time_counter}",grid_vel)
        numpy.save(f"{folder_name}\\part_data\\vel_{time_counter}",fluid.vel.to_numpy()[0:fluid.part_num[None],:])
        print(fluid.vel.to_numpy()[0:fluid.part_num[None],:].shape)
        numpy.save(f"{folder_name}\\part_data\\pos_{time_counter}",fluid.pos.to_numpy()[0:fluid.part_num[None],:])
        print(fluid.pos.to_numpy()[0:fluid.part_num[None],:].shape)

        '''sph steps'''
        frame_div_iter=0
        frame_incom_iter=0
        while time_count<time_counter/refreshing_rate:
            cfl_condition(fluid)
            time_count += dt[None]
            # if time_count >1.1 and flg:
            #     fluid.push_2d_cube(center_pos=[0, 1.3], size=[0.8, 0.8], volume_frac=[1,0], color=0xA21212)
            #     flg=False
            sph_step()
            frame_div_iter+=div_iter_count
            frame_incom_iter+=incom_iter_count
        time_counter += 1
        # statistic(fluid)
        SPH_update_color(fluid)
else:
    """3d only output ply"""
    time_count = float(0)
    time_counter = int(0)
    print('fluid particle count: ', fluid.part_num[None])
    print('bound particle count: ', bound.part_num[None])

    while time_count < 60:
        """ computation loop """
        cfl_condition(fluid)
        time_count += dt[None]
        sph_step()

        """ recording """
        if time_count*refreshing_rate > time_counter:
            time_counter += 1
            print('current time: ', time_count)
            print('time step: ', dt[None])
            SPH_update_color(fluid)
            write_ply(path='ply_3d/fluid_pos', frame_num=time_counter, num=fluid.part_num[None], dim=dim, pos=fluid.pos.to_numpy())

