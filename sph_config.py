import taichi as ti
import numpy as np
import math

# ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32)
ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32, device_memory_GB=1, excepthook=True)

dim = 2
phase_num = 2
init_part_size = 0.2
part_radii_relax = 2
cs = 100
wc_gamma = 7
dynamic_viscosity = np.float32(1e-2)
fluid_part_num = int(2e4)
bound_part_num = int(2e4)
max_part_num = fluid_part_num + bound_part_num
node_num = int(1)
neighb_range = int(1) # range to search neighbors, e.g. when neighb_range=1, search neighbors in (x-1, x, x+1)
gui_res_0 = 1080
divergence_threshold = 1e-3
compression_threshold = 1e-4
iter_threshold_min = 2
iter_threshold_max = 100
refreshing_rate = 30 # frames per second
relaxing_factor=1.01
cfl_factor = 0.5
init_fbm_diffusion_term = 0.000
init_fbm_convection_term = 50
surface_tension_gamma = 10
use_VF = False

np_phase_rest_density = np.empty(shape=phase_num, dtype=np.float32)
np_sim_space_lb = np.empty(shape=dim, dtype=np.float32)
np_sim_space_rt = np.empty(shape=dim, dtype=np.float32)
np_part_size = np.empty(shape=5, dtype=np.float32)
np_sph_h = np.empty(shape=5, dtype=np.float32)
np_sph_sig = np.empty(shape=4, dtype=np.float32)
np_gravity = np.empty(shape=dim, dtype=np.float32)
np_node_dim = np.empty(shape=dim, dtype=np.int32)
np_node_dim_coder = np.empty(shape=dim, dtype=np.int32)
np_neighb_template = np.empty(
    shape=(dim, (neighb_range*2+1)**dim), dtype=np.int32)
np_neighb_dice = np.empty(shape=neighb_range*2+1, dtype=np.int32)

np_phase_rest_density[0] = 10**dim
np_phase_rest_density[1] = 10**dim/2

np_sim_space_lb.fill(-3.5)
np_sim_space_rt.fill(3.5)

np_part_size[1] = init_part_size
np_part_size[2] = math.pow(np_part_size[1], 2)
np_part_size[3] = math.pow(np_part_size[1], 3)
np_part_size[4] = math.pow(np_part_size[1], 4)
np_sph_h[1] = np_part_size[1]*2
np_sph_h[2] = math.pow(np_sph_h[1], 2)
np_sph_h[3] = math.pow(np_sph_h[1], 3)
np_sph_h[4] = math.pow(np_sph_h[1], 4)
np_sph_sig[3] = 8 / math.pi / np_sph_h[1]**3
np_sph_sig[2] = 40 / 7 / math.pi / np_sph_h[1]**2
np_gravity.fill(0)
np_gravity[1] = -9.8
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
