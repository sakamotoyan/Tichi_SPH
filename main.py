from sph import *
import json

""" init data structure """
ngrid = Ngrid()
fluid = Fluid(max_part_num=fluid_part_num)
bound = Fluid(max_part_num=bound_part_num)
gridNodes = Fluid(max_part_num=fluid_part_num)

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
# assign_phase_color(0xA21212,0)
# assign_phase_color(0xffffff,1)

""" setup scene """
lb = np.zeros(dim, np.float32)
rt = np.zeros(dim, np.float32)
mask = np.ones(dim, np.int32)
volume_frac = np.zeros(phase_num, np.float32)
""" push cube """
fluid.push_2d_cube(center_pos=[-0.5, 0], size=[0.7, 1.6], volume_frac=[1,0], color=0x6F7DBC)
fluid.push_2d_cube([0.5,0],[0.7, 1.6],[0,1],0xeacd76)
bound.push_2d_cube([0,0],[2,2],[1,0],0xaaaaaa,4)
""" push gridNodes with no relaxing factor"""
tmp_relaxing_factor,relaxing_factor=relaxing_factor,1
gridNodes.push_2d_cube([0,0],init_part_size * np.floor((np_sim_space_rt-np_sim_space_lb)/init_part_size),[1,0],0x000000)
relaxing_factor=tmp_relaxing_factor

# fluid.push_2d_cube(center_pos=[-1, 0], size=[1.8, 3.6], volume_frac=[1,0], color=0x6F7DBC)
# fluid.push_2d_cube([1,0],[1.8, 3.6],[0,1],0xeacd76)
# bound.push_2d_cube([0,0],[4,4],[1,0],0xaaaaaa,4)

# #fluid.push_2d_cube(center_pos=[0, 2], size=[0.8, 0.8], volume_frac=[1,0], color=0xA21212)
# fluid.push_2d_cube([0,-1],[4-init_part_size*8*1.1, 4-init_part_size*8*1.1],[0,1],0xffffff)
# bound.push_2d_cube([0,0],[4,6],[1,0],0x145b7d,4)

def sph_step():
    global div_iter_count, incom_iter_count
    """ neighbour search """
    ngrid.clear_node()
    ngrid.encode(fluid)
    ngrid.encode(bound)
    ngrid.encode(gridNodes)
    ngrid.mem_shift()
    ngrid.fill_node(fluid)
    ngrid.fill_node(bound)
    ngrid.fill_node(gridNodes)
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
    map_velocity(ngrid, gridNodes,fluid)
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

def write_full_json():
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
    with open("part_json\\"+ ("VF" if use_VF else "DF") +"_" +str(surface_tension_gamma)+"_"+ str(time_counter) + ".json","w") as f:
        f.write(s)

""" GUI system """
time_count = float(0)
time_counter = int(0)
step_counter = int(0)
frame_div_iter=0
frame_incom_iter=0
flg = True
print('fluid particle count: ', fluid.part_num[None])
print('bound particle count: ', bound.part_num[None])
gui = ti.GUI('SPH', to_gui_res(gui_res_0))
while gui.running and not gui.get_event(gui.ESCAPE) and time_counter<250:
    gui.clear(0xffffff)
    frame_div_iter=0
    frame_incom_iter=0
    while time_count*refreshing_rate < time_counter:
        cfl_condition(fluid)
        time_count += dt[None]
        step_counter+=1
        # if time_count >1.1 and flg:
        #     fluid.push_2d_cube(center_pos=[0, 1.3], size=[0.8, 0.8], volume_frac=[1,0], color=0xA21212)
        #     flg=False
        sph_step()
        #print(dt[None],time_count)
        frame_div_iter+=div_iter_count
        frame_incom_iter+=incom_iter_count
        #write_json()
    time_counter += 1
    print('current time: ', time_count)
    print('time step: ', dt[None])
    # statistic(fluid)
    SPH_update_color(fluid)
    gui.circles(to_gui_pos(fluid), radius=to_gui_radii(part_radii_relax), color=to_gui_color(fluid))
    gui.circles(to_gui_pos(bound), radius=to_gui_radii(part_radii_relax), color=to_gui_color(bound))
    gui.circles(to_gui_pos(gridNodes), radius=to_gui_radii(part_radii_relax*0.1), color=to_gui_color(gridNodes))
    gui.show(f"D:\\workspace\\Tichi_SPH\\Tichi_SPH\\img\\rf{refreshing_rate}_{'VF'if use_VF else 'DF'}_{surface_tension_gamma}_{time_counter}.png")
    #write_full_json()
    print("div iter:",frame_div_iter,",frame iter:",frame_incom_iter)
    print(gridNodes.vel.to_numpy())