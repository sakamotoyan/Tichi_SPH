import taichi as ti
import numpy as np
import math
import sys
import getopt
import json
from plyfile import *


################################## Tools ##################################################
def read_param(param, paramname):
    if param is None:
        raise Exception("The parameter " + paramname + " is missing, or invalid parameter")
    return param


def read_array_param(param, default):
    if param is None:
        return default
    elif len(param) != len(default):
        raise Exception('error: wrong array dimension in config file ', param)
    return param


'''parse command line'''
config_file = ""
scenario_file = ""
try:
    opts, args = getopt.getopt(sys.argv[1:], "c:s:", [
                               "configFile=", "scenarioFile="])
except getopt.GetoptError:
    print('main.py -c <configfile> -s <scenariofile>')
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-c", "--configFile"):
        config_file = arg
    elif opt in ("-s", "--secnarioFile"):
        scenario_file = arg
################################## END Tools #############################################
#
#
################################## Read json files #######################################
'''read config file'''
config_buffer = {}
scenario_buffer = {}
try:
    config_buffer = json.load(open(config_file))
except Exception:
    print('no config file or config file invalid')
    exit()
try:
    scenario_buffer = json.load(open(scenario_file))
except Exception:
    print('no scenario file or scenario file invalid')
    exit()
################################## END Read json files ####################################
#
#
################################## Init Taichi ############################################
config_ti_arch = config_buffer.get('arch')
config_device_mem = config_buffer.get('device_memory_GB')
if(config_ti_arch == 'cpu'):
    ti.init(arch=eval('ti.'+config_ti_arch),
            default_fp=ti.f32, default_ip=ti.i32)
elif(config_ti_arch == 'gpu'):
    ti.init(arch=eval('ti.'+config_ti_arch),
            device_memory_GB=config_device_mem, default_fp=ti.f32, default_ip=ti.i32)
else:
    print('invalid taichi init config!')
    exit(0)
################################## END Init Taichis ######################################
#
#

sim_dim = read_param(scenario_buffer['sim_env']['sim_dim'], 'sim_dim')  # simulation dimensions
phase_num = read_param(scenario_buffer['sim_env']['phase_num'], 'phase_num')  # number of phases
solver_type = read_param(config_buffer.get('solver_type'), 'solver_type')
neighb_range = read_param(config_buffer.get('solver_neighb_range'), 'solver_neighb_range')


@ti.data_oriented
class Config:
    def __init__(self):
        # sim_env
        self.dim = ti.field(int, ())
        self.part_size = ti.field(float, 5)  # particle diameter
        self.fluid_max_part_num = ti.field(int, ())
        self.bound_max_part_num = ti.field(int, ())
        self.sim_space_lb = ti.Vector.field(sim_dim, float, ())  # min coordination of simulation space
        self.sim_space_rt = ti.Vector.field(sim_dim, float, ())  # max coordination of simulation space
        self.dynamic_viscosity = ti.field(float, ())
        self.gravity = ti.Vector.field(sim_dim, float, ())
        self.phase_num = ti.field(int, ())
        self.phase_rest_density = ti.Vector.field(phase_num, float, ())  # rest density of each phase
        self.phase_rgb = ti.Vector.field(3, float, phase_num)

        # solver
        self.dt = ti.field(float, ())
        # kernel function
        self.kernel_h = ti.field(float, 5)
        self.kernel_sig = ti.field(float, 4)  # normalizer for kernel W for different dimensions

        # DFSPH
        self.wc_gamma = ti.field(int, ())
        self.divergence_threshold = ti.field(float, ())  # error threshold for divergence free solver
        self.compression_threshold = ti.field(float, ())  # error threshold for divergence free solver
        self.iter_threshold_min = ti.field(int, ())  # min iterations for solvers
        self.iter_threshold_max = ti.field(int, ())  # max iterations for solvers

        # CFL
        self.if_cfl = ti.field(int, ())  # 1: use cfl, 0: do not use cfl
        self.cs = ti.field(int, ())
        self.cfl_factor = ti.field(float, ())  # cfl multiplier

        # neighb
        self.neighb_range = ti.field(int, ())  # range to search neighbors, e.g. when neighb_range=1, search neighbors in (x-1, x, x+1)
        self.neighb_grid_size_TO_global_part_size = ti.field(int, ())

        self.node_num = ti.field(int, ())  # number of neighbor nodes
        self.neighb_grid_size = ti.Vector.field(sim_dim, float, ())
        self.neighb_grid_coder = ti.Vector.field(sim_dim, int, ())
        self.neighb_search_template = ti.Vector.field(sim_dim, int, (neighb_range * 2 + 1) ** sim_dim)
        self.private_neighb_dice = ti.field(int, neighb_range * 2 + 1)

        # FBM
        self.fbm_diffusion_term = ti.field(float, ())
        self.fbm_convection_term = ti.field(float, ())
        self.surface_tension_gamma = ti.field(float, ())

        # GUI
        self.gui_res = ti.Vector.field(2, int, ())
        self.gui_part_zoomer = ti.field(float, ())  # for gui visualization particle size
        self.gui_fps = ti.field(int, ())
        self.gui_camera_pos = ti.Vector.field(3, float, ())
        self.gui_camera_lookat = ti.Vector.field(3, float, ())
        self.gui_canvas_bgcolor = ti.Vector.field(3, float, ())

        self.init()


    def init_sim_env(self):
        self.dim[None] = sim_dim
        self.part_size[1] = read_param(scenario_buffer['sim_env']['global_part_size'], 'global_part_size')
        self.dynamic_viscosity[None] = read_param(scenario_buffer['sim_env']['global_dynamic_viscosity'], 'global_dynamic_viscosity')
        self.sim_space_lb[None] = read_param(scenario_buffer['sim_env']['sim_space_lb'], 'sim_space_lb')
        self.sim_space_rt[None] = read_param(scenario_buffer['sim_env']['sim_space_rt'], 'sim_space_rt')
        self.gravity[None] = read_param(scenario_buffer['sim_env']['gravity'], 'gravity')
        self.phase_num[None] = phase_num
        self.phase_rest_density[None] = read_param(scenario_buffer['sim_env']['phase_rest_density'], 'phase_rest_density')
        self.fluid_max_part_num[None] = int(read_param(scenario_buffer['fluid']['max_part_num'], 'fluid_max_part_num'))
        self.bound_max_part_num[None] = int(read_param(scenario_buffer['bound']['max_part_num'], 'bound_max_part_num'))

        self.part_size[2] = math.pow(self.part_size[1], 2)
        self.part_size[3] = math.pow(self.part_size[1], 3)
        self.part_size[4] = math.pow(self.part_size[1], 4)

    def init_solver(self):
        # DFSPH
        self.wc_gamma[None] = read_param(config_buffer.get('solver_wc_gamma'), 'solver_wc_gamma')
        self.divergence_threshold[None] = read_param(config_buffer.get('solver_divergence_threshold'), 'solver_divergence_threshold')
        self.compression_threshold[None] = read_param(config_buffer.get('solver_compression_threshold'), 'solver_compression_threshold')
        self.iter_threshold_min[None] = read_param(config_buffer.get('solver_iter_threshold_min'), 'solver_iter_threshold_min')
        self.iter_threshold_max[None] = read_param(config_buffer.get('solver_iter_threshold_max'), 'solver_iter_threshold_max')

        # CFL
        self.if_cfl[None] = read_param(config_buffer.get('solver_if_cfl'), 'solver_if_cfl')
        self.cs[None] = read_param(config_buffer.get('solver_cs'), 'solver_cs')
        self.cfl_factor[None] = read_param(config_buffer.get('solver_cfl_factor'), 'solver_cfl_factor')

        # neighb
        self.neighb_range[None] = read_param(config_buffer.get('solver_neighb_range'), 'solver_neighb_range')
        self.neighb_grid_size_TO_global_part_size[None] = read_param(config_buffer.get('solver_neighb_grid_size_TO_global_part_size'), 'solver_neighb_grid_size_TO_global_part_size')

        # FBM
        self.fbm_diffusion_term[None] = read_param(config_buffer.get('solver_fbm_diffusion_term'), 'solver_fbm_diffusion_term')
        self.fbm_convection_term[None] = read_param(config_buffer.get('solver_fbm_convection_term'), 'solver_fbm_convection_term')
        self.surface_tension_gamma[None] = read_param(config_buffer.get('solver_surface_tension_gamma'), 'solver_surface_tension_gamma')

    def init_gui(self):
        self.gui_res[None] = read_param(config_buffer.get('gui_res'), 'gui_res')
        self.gui_part_zoomer[None] = read_param(config_buffer.get('gui_part_zoomer'), 'gui_part_zoomer')
        self.gui_fps[None] = read_param(config_buffer.get('gui_fps'), 'gui_fps')
        self.gui_canvas_bgcolor[None] = read_param(config_buffer.get('gui_canvas_bgcolor'), 'gui_canvas_bgcolor')
        if sim_dim == 3:
            self.gui_camera_pos[None] = read_param(config_buffer.get('gui_camera_pos'), 'gui_camera_pos')
            self.gui_camera_lookat[None] = read_param(config_buffer.get('gui_camera_lookat'), 'gui_camera_lookat')

    def calculate_kernel_param(self):
        self.kernel_h[1] = self.part_size[1] * 2
        self.kernel_h[2] = math.pow(self.kernel_h[1], 2)
        self.kernel_h[3] = math.pow(self.kernel_h[1], 3)
        self.kernel_h[4] = math.pow(self.kernel_h[1], 4)
        self.kernel_sig[3] = 8 / math.pi / self.kernel_h[1] ** 3
        self.kernel_sig[2] = 40 / 7 / math.pi / self.kernel_h[1] ** 2

    @ti.kernel
    def init_neighb_param(self):
        self.node_num[None] = 1
        self.neighb_grid_size[None] = ti.ceil(
            (self.sim_space_rt[None] - self.sim_space_lb[None]) / self.kernel_h[1])  # need ti.kernel

    def calculate_neighb_param(self):
        self.neighb_grid_coder.fill(1)
        for i in range(sim_dim):
            self.neighb_grid_coder[None][i] = self.node_num[None]
            self.node_num[None] *= int(self.neighb_grid_size[None][i])

        for i in range(self.private_neighb_dice.shape[0]):
            self.private_neighb_dice[i] = - neighb_range + i

        for j in range(self.neighb_search_template.shape[0]):
            tmp = j
            for d in range(sim_dim):
                digit = tmp // (self.private_neighb_dice.shape[0] ** (sim_dim - d - 1))
                tmp = tmp % (self.private_neighb_dice.shape[0] ** (sim_dim - d - 1))
                self.neighb_search_template[j][sim_dim - d - 1] = self.private_neighb_dice[digit]

    def init(self):
        self.init_sim_env()
        self.init_solver()
        self.init_gui()
        self.calculate_kernel_param()
        self.init_neighb_param()
        self.calculate_neighb_param()


config = Config()






