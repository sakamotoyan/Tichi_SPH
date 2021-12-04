import numpy
from sph import *
import json
import time

""" init data structure """
ngrid = Ngrid()
fluid = Fluid(max_part_num=config.fluid_max_part_num[None])
bound = Fluid(max_part_num=config.bound_max_part_num[None])
# grid = Grid()


# config.dt[None] = 0.0005
def init_scenario():
    # init phase color
    for i in range(config.phase_num[None]):
        assign_phase_color(int(scenario_buffer['sim_env']['phase_color_hex'][i], 16), i)

    """ setup scene """
    try:
        for part in scenario_buffer:
            obj = None
            if part == 'fluid':
                obj = fluid
            elif part == 'bound':
                obj = bound
            if obj is not None:
                for param in scenario_buffer[part]['objs']:
                    if param['type'] == 'cube':
                        obj.scene_add_cube(param['start_pos'], param['end_pos'], param['volume_frac'], param['vel'],
                                           int(param['color'], 16), param['particle_relaxing_factor'])
                    elif param['type'] == 'box':
                        obj.scene_add_box(param['start_pos'], param['end_pos'], param['layers'], param['volume_frac'],
                                          param['vel'], int(param['color'], 16), param['particle_relaxing_factor'])
                    elif param['type'] == 'ply':
                        verts = read_ply(param['file_name'])
                        obj.push_part_from_ply(len(verts), verts, param['volume_frac'], param['vel'],
                                               int(param['color'], 16))
    except Exception:
        print('no scenario file or scenario file invalid')
        exit(0)

    # for ggui
    set_unused_par(fluid)
    set_unused_par(bound)


##################################### Write Json ############################################
def write_scene_data():
    data = {
        'grid_count': int(grid.size),
        'grid_lb': [float(grid.lb[i]) for i in range(len(grid.lb))],
        'init_part_size': config.part_size[1],
        'fluid_part_count': fluid.part_num[None],
        'bound_part_count': bound.part_num[None],
        'dt': config.dt[None],
        'frame_rate': config.gui_fps[None]
    }
    json.dump(data, open(f"{solver_type}\\data.json", "w"))


def write_json():
    data = {
        "step": step_counter,
        "frame": time_counter,
        "timeInSimulation": time_count,
        "timeStep": config.dt[None],
        "fps": config.gui_fps[None],
        "iteration": {
            "divergenceFree_iteration": div_iter_count,
            "incompressible_iteration": incom_iter_count,
            "sum_iteration": div_iter_count + incom_iter_count
        },
        "energy": {
            "statistics_kinetic_energy": fluid.statistics_kinetic_energy[None],
            "statistics_gravity_potential_energy": fluid.statistics_gravity_potential_energy[None],
            "sum_energy": fluid.statistics_kinetic_energy[None] + fluid.statistics_gravity_potential_energy[None]
        }
    }
    s = json.dumps(data)
    with open("json\\" + solver_type + str(step_counter) + ".json", "w") as f:
        f.write(s)


def write_full_json(fname):
    global frame_div_iter, frame_incom_iter
    data = {
        "step": step_counter,
        "frame": time_counter,
        "timeInSimulation": time_count,
        "timeStep": config.dt[None],
        "fps": config.gui_fps[None],
        "iteration": {
            "divergenceFree_iteration": frame_div_iter,
            "incompressible_iteration": frame_incom_iter,
            "sum_iteration": frame_div_iter + frame_incom_iter
        },
        "energy": {
            "statistics_kinetic_energy": fluid.statistics_kinetic_energy[None],
            "statistics_gravity_potential_energy": fluid.statistics_gravity_potential_energy[None],
            "sum_energy": fluid.statistics_kinetic_energy[None] + fluid.statistics_gravity_potential_energy[None]
        }
    }
    s = json.dumps(data)
    with open(fname, "w") as f:
        f.write(s)


################################## End Write Json ############################################

###################################### SPH SOLVER ############################################
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
    while div_iter_count < config.iter_threshold_min[None] or fluid.compression[None] > config.divergence_threshold[None]:
        IPPE_adv_psi_init(fluid)
        # IPPE_adv_psi_init(bound)
        IPPE_adv_psi(ngrid, fluid, fluid)
        IPPE_adv_psi(ngrid, fluid, bound)
        # IPPE_adv_psi(ngrid, bound, fluid)
        IPPE_psi_adv_non_negative(fluid)
        # IPPE_psi_adv_non_negative(bound)
        IPPE_update_vel_adv(ngrid, fluid, fluid)
        IPPE_update_vel_adv(ngrid, fluid, bound)
        div_iter_count += 1
        if div_iter_count > config.iter_threshold_max[None]:
            break
    SPH_vel_adv_2_vel(fluid)
    """ SPH advection """
    SPH_advection_gravity_acc(fluid)
    SPH_advection_viscosity_acc(ngrid, fluid, fluid)
    SPH_advection_surface_tension_acc(ngrid, fluid, fluid)
    SPH_advection_update_vel_adv(fluid)
    """ IPPE SPH pressure """
    incom_iter_count = 0
    while incom_iter_count < config.iter_threshold_min[None] or fluid.compression[None] > config.compression_threshold[None]:
        IPPE_adv_psi_init(fluid)
        # IPPE_adv_psi_init(bound)
        IPPE_adv_psi(ngrid, fluid, fluid)
        IPPE_adv_psi(ngrid, fluid, bound)
        # IPPE_adv_psi(ngrid, bound, fluid)
        IPPE_psi_adv_non_negative(fluid)
        # IPPE_psi_adv_non_negative(bound)
        IPPE_update_vel_adv(ngrid, fluid, fluid)
        IPPE_update_vel_adv(ngrid, fluid, bound)
        incom_iter_count += 1
        if incom_iter_count > config.iter_threshold_max[None]:
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
    # map_velocity(ngrid, grid, fluid)
    return div_iter_count, incom_iter_count
    """ SPH debug """
#################################### END SPH SOLVER ###########################################

init_scenario()
# write_scene_data()

# show_window = False
show_window = True
show_bound = False
show_help = True
show_run_info = True
op_system_run = False
op_write_file = False

# for time record
is_first_time = True
time_real = 0
time_start = 0

time_count = float(0)
time_counter = int(0)
step_counter = int(0)
frame_div_iter = 0
frame_incom_iter = 0
div_iter_count = 0
incom_iter_count = 0


def show_options():
    global show_bound, show_help, show_run_info
    global op_system_run, op_write_file

    # run info
    if show_run_info:
        window.GUI.begin("time info", 0.05, 0.05, 0.2, 0.2)
        window.GUI.text("fluid particle count: " + str(fluid.part_num[None]))
        window.GUI.text("bound particle count: " + str(bound.part_num[None]))
        window.GUI.text("simulation time: " + str('%.3f' % time_count))
        window.GUI.text("real time: " + str('%.3f' % time_real))
        window.GUI.text("time step: " + str('%.3f' % config.dt[None]))
        window.GUI.end()

    if show_help:
        window.GUI.begin("options", 0.05, 0.3, 0.2, 0.2)
        window.GUI.text("h: help")
        window.GUI.text("w: front")
        window.GUI.text("s: back")
        window.GUI.text("a: left")
        window.GUI.text("d: right")
        window.GUI.text("RMB: rotate")
        window.GUI.text("b: display boundary")
        window.GUI.text("r: run system")
        window.GUI.text("f: write file")
        window.GUI.end()

    if window.get_event(ti.ui.PRESS):
        # run
        if window.event.key == 'r':
            op_system_run = not op_system_run
            print("start to run:", op_system_run)

        if window.event.key == 'f':
            op_write_file = not op_write_file
            print("write file:", op_write_file)

        if window.event.key == 'b':
            show_bound = not show_bound
            print("show boundary:", show_bound)

        if window.event.key == 'i':
            show_run_info = not show_run_info
            print("show run information:", show_run_info)

        if window.event.key == 'h':
            show_help = not show_help
            print("show help:", show_help)


def render3d():
    canvas.set_background_color(background_color)
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light(ambient_color)
    # Configuring light sources, must set one plint light, otherwise occurs an error (seem to be a bug)
    scene.point_light(pos=(2, 1.5, -1.5), color=(0.8, 0.8, 0.8))

    update_color_vector(fluid)
    update_color_vector(bound)
    scene.particles(fluid.pos, per_vertex_color=fluid.color_vector, radius=dispaly_radius)
    if show_bound:
        scene.particles(bound.pos, per_vertex_color=bound.color_vector, radius=dispaly_radius)

    canvas.scene(scene)  # Render the scene


def render2d():
    canvas.set_background_color(background_color)
    update_color_vector(fluid)
    update_color_vector(bound)
    to_gui_pos(fluid)
    to_gui_pos(bound)
    canvas.circles(fluid.gui_2d_pos, per_vertex_color=fluid.color_vector,
                   radius=to_gui_radii(config.gui_part_zoomer[None]))
    if show_bound:
        canvas.circles(bound.gui_2d_pos, per_vertex_color=bound.color_vector,
                       radius=to_gui_radii(config.gui_part_zoomer[None]))


def run_step():
    global time_counter, time_count, frame_div_iter, frame_incom_iter
    global time_real, time_start, is_first_time
    time_counter += 1

    '''according fps to render'''
    while time_count < time_counter / config.gui_fps[None]:
        if is_first_time:
            time_start = time.time()
            is_first_time = False
        """ computation loop """
        cfl_condition(fluid)
        time_count += config.dt[None]
        sph_step()
        frame_div_iter += div_iter_count
        frame_incom_iter += incom_iter_count
        print('current time: ', time_count)
        # print('real time: ', time_real)
        print('time step: ', config.dt[None])

    ''''render after one sph step'''
    # if is_first_time:
    #     time_start = time.time()
    #     is_first_time = False
    # """ computation loop """
    # cfl_condition(fluid)
    # time_count += config.dt[None]
    # sph_step()
    # frame_div_iter += div_iter_count
    # frame_incom_iter += incom_iter_count

    time_real = time.time() - time_start
    print('current time: ', time_count)
    print('---------------------real time------------------------', time_real)
    print('time step: ', config.dt[None])
    SPH_update_color(fluid)


def write_files():
    write_full_json(f"{solver_type}\\json\\" + "frame" + str(time_counter) + ".json")
    # numpy.save(f"{solver_type}\\grid_data\\vel_{time_counter}", grid.vel.to_numpy())
    numpy.save(f"{solver_type}\\part_data\\vel_{time_counter}", fluid.vel.to_numpy()[0:fluid.part_num[None], :])
    numpy.save(f"{solver_type}\\part_data\\pos_{time_counter}", fluid.pos.to_numpy()[0:fluid.part_num[None], :])


print('fluid particle count: ', fluid.part_num[None])
print('bound particle count: ', bound.part_num[None])
if show_window:
    """define window, canvas, scene and camera"""
    window = ti.ui.Window("Fluid Simulation", (config.gui_res[None][0], config.gui_res[None][1]), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.make_camera()
    camera.position(config.gui_camera_pos[None][0], config.gui_camera_pos[None][1], config.gui_camera_pos[None][2])
    camera.lookat(config.gui_camera_lookat[None][0], config.gui_camera_lookat[None][1], config.gui_camera_lookat[None][2])
    camera.fov(55)
    background_color = (
    (config.gui_canvas_bgcolor[None][0], config.gui_canvas_bgcolor[None][1], config.gui_canvas_bgcolor[None][2]))
    ambient_color = (0.7, 0.7, 0.7)
    dispaly_radius = config.part_size[1] * 0.5  # render particle size

    # print('grid count:', grid.size)
    # numpy.save(f"{solver_type}\\grid_data\\pos",grid.pos.to_numpy())
    if dim == 2:
        while window.running:
            render2d()
            show_options()
            if op_system_run:
                run_step()
            if op_write_file:
                print("--------------")
                write_files()
                window.write_image(f"{solver_type}\\img\\rf{int(config.gui_fps[None] + 1e-5)}_{time_counter}.png")
            window.show()
    else:
        while window.running:
            render3d()
            show_options()
            if op_system_run:
                run_step()
            if op_write_file:
                write_ply(path='ply_3d/fluid_pos', frame_num=time_counter, dim=dim, num=fluid.part_num[None], pos=fluid.pos.to_numpy())
                window.write_image(f"{solver_type}\\img\\rf{int(config.gui_fps[None] + 1e-5)}_{time_counter}.png")
                write_files()
            window.show()

else:
    if dim == 2:
        while time_count < 60:
            run_step()
            if op_write_file:
                write_files()
    else:
        while time_count < 60:
            run_step()
            if op_write_file:
                # write_files()
                write_ply(path='ply_3d/fluid_pos', frame_num=time_counter, num=fluid.part_num[None], dim=dim, pos=fluid.pos.to_numpy())
