from sph import *

""" init data structure """
ngrid = Ngrid()
fluid = Fluid(max_part_num=fluid_part_num)
bound = Fluid(max_part_num=bound_part_num)
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

assign_phase_color(0xffffff,0)
assign_phase_color(0x0000ff,1)

""" setup scene """
lb = np.zeros(dim, np.float32)
rt = np.zeros(dim, np.float32)
mask = np.ones(dim, np.int32)
volume_frac = np.zeros(phase_num, np.float32)
""" push cube """
fluid.push_2d_cube(center_pos=[-1, 0], size=[1.8, 3.6], volume_frac=[1,0], color=0x068587)
fluid.push_2d_cube([1,0],[1.8, 3.6],[0,1],0x8f0000)
bound.push_2d_cube([0,0],[4,4],[1,0],0xFF4500,4)

def sph_step():
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
    SPH_update_volume_frac(fluid)
    SPH_update_mass(fluid)
    SPH_update_pos(fluid)
    return div_iter_count, incom_iter_count
    """ SPH debug """

""" GUI system """
time_count = float(0)
time_counter = int(0)
print('fluid particle count: ', fluid.part_num[None])
print('bound particle count: ', bound.part_num[None])
gui = ti.GUI('SPH', to_gui_res(gui_res_0))
while gui.running and not gui.get_event(gui.ESCAPE):
    gui.clear(0x112F41)
    while time_count*refreshing_rate < time_counter:
        cfl_condition(fluid)
        time_count += dt[None]
        sph_step()
    time_counter += 1
    print('current time: ', time_count)
    print('time step: ', dt[None])
    # statistic(fluid)
    SPH_update_color(fluid)
    gui.circles(to_gui_pos(fluid), radius=to_gui_radii(part_radii_relax), color=to_gui_color(fluid))
    gui.circles(to_gui_pos(bound), radius=to_gui_radii(part_radii_relax), color=to_gui_color(bound))
    gui.show(f"img\\{time_counter}_rf{refreshing_rate}.png")