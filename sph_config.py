import taichi as ti
import numpy as np
import math
import sys
import getopt
import json
from plyfile import *


def read_param(param,default):
    return param if param is not None else default

def read_array_param(param,default):
    if param is None:
        return default
    elif len(param)!=len(default):
        raise Exception('error: wrong array dimension in config file ',param)
    return param

'''parse command line'''
config_file=""
scenario_file=""
try:
    opts, args = getopt.getopt(sys.argv[1:],"c:s:",["configFile=","scenarioFile="])
except getopt.GetoptError:
    print('main.py -c <configfile> -s <scenariofile>')
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-c", "--configFile"):
        config_file = arg
    elif opt in ("-s", "--secnarioFile"):
        scenario_file = arg

'''read config file'''
config = {}
try:
    config = json.load(open(config_file))
except Exception:
    print('no config file or config file invalid, use default config values')

'''init default parameters'''
#memory allocation
device_memory_GB  = read_param(config.get('device_memory_GB'),default=9) # memory to allocate for tichi
fluid_part_num  = int(read_param(config.get('fluid_part_num'),default=2e6)) # max number of fluid particles
bound_part_num  = int(read_param(config.get('bound_part_num'),default=2e5)) # max number of boundary particles

#simulation
dim  = read_param(config.get('dim'),default=2) # simulation dimensions
init_part_size  = read_param(config.get('init_part_size'),default=0.04) # particle diameter
dynamic_viscosity  = np.float32(read_param(config.get('dynamic_viscosity'),default=1e-2)) # dynamic viscosity
wc_gamma  = read_param(config.get('wc_gamma'),default=7) # for WCSPH
divergence_threshold  = read_param(config.get('divergence_threshold'),default=1e-3) # error threshold for divergence free solver
compression_threshold  = read_param(config.get('compression_threshold'),default=1e-4) # error threshold for divergence free solver
iter_threshold_min  = read_param(config.get('iter_threshold_min'),default=2) # minimal iterations for solvers
iter_threshold_max  = read_param(config.get('iter_threshold_max'),default=100) # max iterations for solvers
cs  = read_param(config.get('cs'),default=100) # for cfl condition
cfl_factor  = read_param(config.get('cfl_factor'),default=0.5) # cfl multiplier
use_VF  = read_param(config.get('use_VF'),default=False) # whether to use VFSPH (use DFSPH if False)

#scenario
relaxing_factor  = read_param(config.get('relaxing_factor'),default=1.01) # spacing relaxing multiplier when initializaing secnario
np_sim_space_lb = np.array(read_array_param(config.get('np_sim_space_lb'),default=[-3.5]*dim), dtype =np.float32) # min coordination of simulation space
np_sim_space_rt = np.array(read_array_param(config.get('np_sim_space_rt'),default=[3.5]*dim), dtype =np.float32) # max coordination of simulation space
np_gravity = np.array(read_array_param(config.get('np_gravity'),default=[(-9.8 if i==1 else 0) for i in range(dim)]), dtype = np.float32) # gravity (default -9.8m/s on y axis)

#neighbor search
neighb_range  = int(read_param(config.get('neighb_range'),default=1)) # range to search neighbors, e.g. when neighb_range=1, search neighbors in (x-1, x, x+1)

#saves
refreshing_rate  = read_param(config.get('refreshing_rate'),default=30) # approximate frames per second

#gui
gui_res_0  = read_param(config.get('gui_res_0'),default=1080) # resolution of gui
part_radii_relax  = read_param(config.get('part_radii_relax'),default=2) # for gui visualization particle size

#multiphase
phase_num  = read_param(config.get('phase_num'),default=2) # number of phases
np_phase_rest_density = np.array(read_array_param(config.get('np_phase_rest_density'),default=[10**dim/(2**i) for i in range(phase_num)]), dtype = np.float32) # rest density of each phase
#multiphase FBM
init_fbm_diffusion_term  = read_param(config.get('init_fbm_diffusion_term'),default=0.000) # mixture model diffusion
init_fbm_convection_term  = read_param(config.get('init_fbm_convection_term'),default=50) # mixture model 

#surface tension
surface_tension_gamma  = read_param(config.get('surface_tension_gamma'),default=1) # surface tension parameter

#grid (for particle-grid mapping)
np_grid_size  = np.array(read_array_param(config.get('np_grid_size'),default=[4]*dim), dtype=np.float32) # coverage of grid
np_grid_lb  = np.array(read_array_param(config.get('np_grid_lb'),default=[-2]*dim),dtype=np.float32) # left-bottom corner of grid
grid_dist  = read_param(config.get('grid_dist'),default=0.04) # spacing of grid

'''calculated parameters'''
max_part_num = fluid_part_num + bound_part_num
#powers of particle size
np_part_size = np.empty(shape=5, dtype=np.float32)
np_part_size[1] = init_part_size
np_part_size[2] = math.pow(np_part_size[1], 2)
np_part_size[3] = math.pow(np_part_size[1], 3)
np_part_size[4] = math.pow(np_part_size[1], 4)
#powers of support radius
np_sph_h = np.empty(shape=5, dtype=np.float32)
np_sph_h[1] = np_part_size[1]*2
np_sph_h[2] = math.pow(np_sph_h[1], 2)
np_sph_h[3] = math.pow(np_sph_h[1], 3)
np_sph_h[4] = math.pow(np_sph_h[1], 4)
#normalizer for kernel W for different dimensions
np_sph_sig = np.empty(shape=4, dtype=np.float32)
np_sph_sig[3] = 8 / math.pi / np_sph_h[1]**3
np_sph_sig[2] = 40 / 7 / math.pi / np_sph_h[1]**2
#neighbor search
node_num = int(1) #number of neighbor nodes
np_node_dim = np.empty(shape=dim, dtype=np.int32)
np_node_dim_coder = np.empty(shape=dim, dtype=np.int32)
np_neighb_template = np.empty(
    shape=(dim, (neighb_range*2+1)**dim), dtype=np.int32)
np_neighb_dice = np.empty(shape=neighb_range*2+1, dtype=np.int32)
np_node_dim[:] = np.ceil(
    (np_sim_space_rt - np_sim_space_lb) / np_sph_h[1]).astype(np.int32)
np_node_dim_coder.fill(1)
for i in range(dim):
    np_node_dim_coder[i] = node_num
    node_num *= int(np_node_dim[i])
for i in range(np_neighb_dice.shape[0]):
    np_neighb_dice[i] = -neighb_range + i
for j in range(np_neighb_template.shape[1]):
    tmp = j
    for d in range(dim):
        digit = tmp // (np_neighb_dice.shape[0]**(dim-d-1))
        tmp = tmp % (np_neighb_dice.shape[0]**(dim-d-1))
        np_neighb_template[dim-d-1][j] = np_neighb_dice[digit]

'''init taichi'''
ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32,
        device_memory_GB=device_memory_GB, excepthook=True)
