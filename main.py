import numpy
from sph import *
import json
import time


""" init data structure """
ngrid = Ngrid()
fluid = Fluid(max_part_num=config.fluid_max_part_num[None])
bound = Fluid(max_part_num=config.bound_max_part_num[None])
grid = Grid()
globalvar = GlobalVariable()
scenario_wait_buffer = [] # scenario objects waiting to be added at a certain time
inlets = [] #stores inlets that pour water in

def get_obj_from_str(s):
    if s == 'fluid':
        return fluid
    elif s == 'bound':
        return bound
    else:
        print('scenario WARNING: object does not exist (name:',s,')')
        return None

#helper function for inlet
def inlet_helper_get_dt(norm,speed,relaxing_factor,config):
    norm=np_normalize(np.array(norm))
    if speed<0:
        norm, speed = -norm, -speed
    acc_vec=config.gravity.to_numpy()
    acc=np.dot(norm,acc_vec)
    r = config.part_size[1]
    if acc < 1e-6:
        if speed > 0:
            return r * relaxing_factor / speed
        else:
            return float("inf")
    else:
        return ((speed ** 2 + 2 * acc * r) ** 0.5 - speed) / acc * relaxing_factor


def add_to_scenario(obj_str,param):
    obj = get_obj_from_str(obj_str)
    if obj is None:
        print('scenario WARNING: obj \'',obj_str,'\' non-exist.')
    else:
        if param['type'] == 'cube':
            obj.scene_add_cube(param['start_pos'], param['end_pos'], param['volume_frac'], param['vel'], int(param['color'], 16), param['particle_relaxing_factor'])
        elif param['type'] == 'box':
            obj.scene_add_box(param['start_pos'], param['end_pos'], param['layers'], param['volume_frac'], param['vel'], int(param['color'], 16), param['particle_relaxing_factor'])
        elif param['type'] == 'ply':
            verts = read_ply(param['file_name'])
            obj.push_part_from_ply(len(verts), verts, param['volume_frac'], param['vel'], int(param['color'], 16))
        elif param['type']=='inlet':
            if np.linalg.norm(np.array(param['norm'])) < 1e-6:
                raise Exception('inlet ERROR: magnitute of attribute \'norm\' should not be zero.')
            param['obj']=obj_str
            param['dt']=inlet_helper_get_dt(param['norm'],param['speed'],param['particle_relaxing_factor'],config)
            param['t_pre']=globalvar.time_count-param['dt']
            inlets.append(param)

def init_scenario():
    """ setup scene """
    try:
        for part in scenario_buffer:
            if part != "sim_env":
                for param in scenario_buffer[part]['objs']:
                    if not 'time' in param or param['time'] <= 0.0: # add now
                        add_to_scenario(part,param)
                    else: #add later
                        param['_part']=part
                        scenario_wait_buffer.append(param)
    except Exception:
        raise Exception('scenario ERROR: no scenario file or scenario file invalid.')

    # for ggui
    set_unused_par(fluid)
    set_unused_par(bound)
    SPH_update_color(fluid)

def refresh_inlets():
    for i in range(len(inlets)-1,-1,-1):
        inlet=inlets[i]
        if 'end_time' in inlet and globalvar.time_count>=inlet['end_time']:
            inlets.pop(i)
        elif globalvar.time_count>=inlet['t_pre']+inlet['dt']:
            try:
                inlet['t_pre']=globalvar.time_count
                get_obj_from_str(inlet['obj']).scene_add_from_inlet(
                    inlet['center'],
                    inlet['size'],
                    inlet['norm'],
                    inlet['speed'],
                    inlet['volume_frac'],
                    int(inlet['color'], 16),
                    inlet['particle_relaxing_factor']
                )
            except Exception:
                warn('inlet WARNING: refresh_inlets failed for:',json.dumps(inlet))
                inlets.pop(i)

def refresh_scenario():
    for i in range(len(scenario_wait_buffer)-1,-1,-1):
        param=scenario_wait_buffer[i]
        try:
            if param['time'] <= globalvar.time_count:
                add_to_scenario(param['_part'],param)
                scenario_wait_buffer.pop(i)
        except Exception:
            warn('scenario WARNING: invalid scenario object:',json.dumps(param))
            scenario_wait_buffer.pop(i)
    refresh_inlets()

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
        "step": globalvar.step_counter,
        "frame": globalvar.time_counter,
        "timeInSimulation": globalvar.time_count,
        "timeStep": config.dt[None],
        "fps": config.gui_fps[None],
        "iteration": {
            "divergenceFree_iteration": globalvar.div_iter_count,
            "incompressible_iteration": globalvar.incom_iter_count,
            "sum_iteration": globalvar.div_iter_count + globalvar.incom_iter_count
        },
        "energy": {
            "statistics_kinetic_energy": fluid.statistics_kinetic_energy[None],
            "statistics_gravity_potential_energy": fluid.statistics_gravity_potential_energy[None],
            "sum_energy": fluid.statistics_kinetic_energy[None] + fluid.statistics_gravity_potential_energy[None]
        }
    }
    s = json.dumps(data)
    with open("json\\" + solver_type + str(globalvar.step_counter) + ".json", "w") as f:
        f.write(s)


def write_full_json(fname):
    data = {
        "step": globalvar.step_counter,
        "frame": globalvar.time_counter,
        "timeInSimulation": globalvar.time_count,
        "timeStep": config.dt[None],
        "fps": config.gui_fps[None],
        "iteration": {
            "divergenceFree_iteration": globalvar.frame_div_iter,
            "incompressible_iteration": globalvar.frame_incom_iter,
            "sum_iteration": globalvar.frame_div_iter + globalvar.frame_incom_iter
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


def sph_step():
    # global div_iter_count, incom_iter_count
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
    globalvar.div_iter_count = 0
    SPH_vel_2_vel_adv(fluid)
    while globalvar.div_iter_count<config.iter_threshold_min[None] or fluid.compression[None]>config.divergence_threshold[None]:
        IPPE_adv_psi_init(fluid)
        # IPPE_adv_psi_init(bound)
        IPPE_adv_psi(ngrid, fluid, fluid)
        IPPE_adv_psi(ngrid, fluid, bound)
        # IPPE_adv_psi(ngrid, bound, fluid)
        IPPE_psi_adv_non_negative(fluid)
        # IPPE_psi_adv_non_negative(bound)
        IPPE_update_vel_adv(ngrid, fluid, fluid)
        IPPE_update_vel_adv(ngrid, fluid, bound)
        globalvar.div_iter_count+=1
        if globalvar.div_iter_count>config.iter_threshold_max[None]:
            break
    SPH_vel_adv_2_vel(fluid)
    """ SPH advection """
    SPH_advection_gravity_acc(fluid)
    SPH_advection_viscosity_acc(ngrid, fluid, fluid)
    SPH_advection_surface_tension_acc(ngrid, fluid, fluid)
    SPH_advection_update_vel_adv(fluid)
    """ IPPE SPH pressure """
    globalvar.incom_iter_count = 0
    while globalvar.incom_iter_count<config.iter_threshold_min[None] or fluid.compression[None]>config.compression_threshold[None]:
        IPPE_adv_psi_init(fluid)
        # IPPE_adv_psi_init(bound)
        IPPE_adv_psi(ngrid, fluid, fluid)
        IPPE_adv_psi(ngrid, fluid, bound)
        # IPPE_adv_psi(ngrid, bound, fluid)
        IPPE_psi_adv_non_negative(fluid)
        # IPPE_psi_adv_non_negative(bound)
        IPPE_update_vel_adv(ngrid, fluid, fluid)
        IPPE_update_vel_adv(ngrid, fluid, bound)
        globalvar.incom_iter_count+=1
        if globalvar.incom_iter_count>config.iter_threshold_max[None]:
            break
    """ debug info """
    # print('iter div: ', globalvar.div_iter_count)
    # print('incom div: ', globalvar.incom_iter_count)
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
    # return globalvar.div_iter_count, globalvar.incom_iter_count
    """ SPH debug """


init_scenario()
# write_scene_data()


def show_options():
    # run info
    if globalvar.show_run_info:
        window.GUI.begin("time info", 0.05, 0.05, 0.2, 0.2)
        window.GUI.text("fluid particle count: " + str(fluid.part_num[None]))
        window.GUI.text("bound particle count: " + str(bound.part_num[None]))
        window.GUI.text("simulation time: " + str('%.7f' % globalvar.time_count))
        window.GUI.text("real time: " + str('%.7f' % globalvar.time_real))
        window.GUI.text("time step: " + str('%.7f' % config.dt[None]))
        window.GUI.end()

    if globalvar.show_help:
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
            globalvar.op_system_run = not globalvar.op_system_run
            print("start to run:", globalvar.op_system_run)

        if window.event.key == 'f':
            globalvar.op_write_file = not globalvar.op_write_file
            print("write file:", globalvar.op_write_file)

        if window.event.key == 'b':
            globalvar.show_bound = not globalvar.show_bound
            print("show boundary:", globalvar.show_bound)

        if window.event.key == 'i':
            globalvar.show_run_info = not globalvar.show_run_info
            print("show run information:", globalvar.show_run_info)

        if window.event.key == 'h':
            globalvar.show_help = not globalvar.show_help
            print("show help:", globalvar.show_help)


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
    if globalvar.show_bound:
        scene.particles(bound.pos, per_vertex_color=bound.color_vector, radius=dispaly_radius)

    canvas.scene(scene)  # Render the scene


def render2d():
    canvas.set_background_color(background_color)
    update_color_vector(fluid)
    update_color_vector(bound)
    to_gui_pos(fluid)
    to_gui_pos(bound)
    canvas.circles(fluid.gui_2d_pos, per_vertex_color=fluid.color_vector, radius=to_gui_radii(config.gui_part_zoomer[None]))
    if globalvar.show_bound:
        canvas.circles(bound.gui_2d_pos, per_vertex_color=bound.color_vector, radius=to_gui_radii(config.gui_part_zoomer[None]))


def run_step():
    globalvar.time_counter += 1

    while globalvar.time_count < globalvar.time_counter / config.gui_fps[None]:
        if globalvar.is_first_time:
            globalvar.time_start = time.time()
            globalvar.is_first_time = False
        """ computation loop """
        refresh_scenario()
        cfl_condition(fluid)
        globalvar.time_count += config.dt[None]
        sph_step()
        globalvar.frame_div_iter += globalvar.div_iter_count
        globalvar.frame_incom_iter += globalvar.incom_iter_count
        # print('current time: ', globalvar.time_count)
        # # print('real time: ', globalvar.time_real)
        # print('time step: ', config.dt[None])

    # if globalvar.is_first_time:
    #     globalvar.time_start = time.time()
    #     globalvar.is_first_time = False
    # """ computation loop """
    # cfl_condition(fluid)
    # globalvar.time_count += config.dt[None]
    # sph_step()
    globalvar.time_real = time.time() - globalvar.time_start
    print('current time: ', globalvar.time_count)
    print('---------------------real time------------------------', globalvar.time_real)
    print('time step: ', config.dt[None])
    SPH_update_color(fluid)


def write_files():
    window.write_image(f"{solver_type}\\img\\rf{int(config.gui_fps[None] + 1e-5)}_{globalvar.time_counter}.png")  # have to call after render2d()/render3d()
    write_ply(path=f'{solver_type}\\ply\\fluid_pos', frame_num=globalvar.time_counter, dim=dim, num=fluid.part_num[None],pos=fluid.pos.to_numpy())

    write_full_json(f"{solver_type}\\json\\" + "frame" + str(globalvar.time_counter) + ".json")
    # numpy.save(f"{solver_type}\\grid_data\\vel_{globalvar.time_counter}", grid.vel.to_numpy())
    numpy.save(f"{solver_type}\\part_data\\vel_{globalvar.time_counter}", fluid.vel.to_numpy()[0:fluid.part_num[None], :])
    numpy.save(f"{solver_type}\\part_data\\pos_{globalvar.time_counter}", fluid.pos.to_numpy()[0:fluid.part_num[None], :])


print('fluid particle count: ', fluid.part_num[None])
print('bound particle count: ', bound.part_num[None])
if globalvar.show_window:
    """define window, canvas, scene and camera"""
    window = ti.ui.Window("Fluid Simulation", (config.gui_res[None][0], config.gui_res[None][1]), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.make_camera()
    camera.position(config.gui_camera_pos[None][0], config.gui_camera_pos[None][1], config.gui_camera_pos[None][2])
    camera.lookat(config.gui_camera_lookat[None][0], config.gui_camera_lookat[None][1], config.gui_camera_lookat[None][2])
    camera.fov(55)
    background_color = ((config.gui_canvas_bgcolor[None][0], config.gui_canvas_bgcolor[None][1], config.gui_canvas_bgcolor[None][2]))
    ambient_color = (0.7, 0.7, 0.7)
    dispaly_radius = config.part_size[1]*0.5  # render particle size

    # print('grid count:', grid.size)
    # numpy.save(f"{solver_type}\\grid_data\\pos",grid.pos.to_numpy())
    if dim == 2:
        while window.running:
            if globalvar.op_system_run:
                run_step()
            show_options()
            render2d()
            if globalvar.op_system_run and globalvar.op_write_file:
                write_files()
            window.show()
    else:
        while window.running:
            if globalvar.op_system_run:
                run_step()
            show_options()
            render3d()
            if globalvar.op_system_run and globalvar.op_write_file:
                write_files()
            window.show()

else:
    while globalvar.time_count < 60:
        run_step()
        if globalvar.op_write_file:
            write_files()




