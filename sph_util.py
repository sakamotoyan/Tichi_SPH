from sph_config import *

######################################## SPH Kernel functions ########################################
@ti.func
def C(r, config: ti.template()):
    q = r / config.kernel_h[1]
    tmp = 0.0
    if q <= 0.5:
        tmp = (2 * (1 - q) ** 3 * q ** 3 - 1 / 64)
    elif q > 0.5 and q < 1:
        tmp = (1 - q) ** 3 * q ** 3
    tmp *= 32 / math.pi / config.kernel_h[1] ** 3
    return tmp


@ti.func
def W(r, config: ti.template()):
    q = r / config.kernel_h[1]
    tmp = 0.0
    if q <= 0.5:
        tmp = 6 * (q ** 3 - q ** 2) + 1
    elif q > 0.5 and q < 1:
        tmp = 2 * (1 - q) ** 3
    tmp *= config.kernel_sig[config.dim[None]]
    return tmp


@ti.func
def W_grad(r, config: ti.template()):
    q = r / config.kernel_h[1]
    tmp = 0.0
    if q <= 0.5:
        tmp = 6 * (3 * q ** 2 - 2 * q)
    elif q > 0.5 and q < 1:
        tmp = -6 * (1 - q) ** 2
    tmp *= config.kernel_sig[config.dim[None]] / config.kernel_h[1]
    return tmp


@ti.func
def W_lap(x_ij: ti.template(), r, V_j, A: ti.template(), config: ti.template()):
    return 2 * (2 + config.dim[None]) * V_j * W_grad(r, config) * x_ij.normalized() * A.dot(x_ij) / (
                0.01 * config.kernel_h[2] + r ** 2)


######################################## Neighb search ########################################

# Input:  pos->ti.Vector([{dim,float}])
# Output: Integer
@ti.func
def node_encode(pos: ti.template(), config: ti.template()):
    return int((pos - config.sim_space_lb[None]) // config.kernel_h[1])

# Input:  dim->ti.Vector([{dim,float}])
# Output: Integer
@ti.func
def dim_encode(dim: ti.template(), config: ti.template()):
    return config.neighb_grid_coder[None].dot(dim)

######################################## Tool functions ########################################

# Input:  original_file_path->string
# Output: trimmed_file_path->string
def trim_path_dir(original_file_path):
    if original_file_path.find('\\') > 0 and original_file_path.find('/') > 0:
        return original_file_path
    elif original_file_path.find('\\') > 0:
        file_path_list = original_file_path.split('\\')
    elif original_file_path.find('/') > 0:
        file_path_list = original_file_path.split('/')
    trimmed_file_path = file_path_list[0]
    for i in range(len(file_path_list)-1):
        trimmed_file_path = os.path.join(trimmed_file_path, file_path_list[i+1])
    return trimmed_file_path

# Input:  vec->ti.Vector([{n,float}])
# Output: Bool
@ti.func
def has_negative(vec: ti.template()):
    is_n = False
    for i in ti.static(range(vec.n)):
        if vec[i] < 0:
            is_n = True
    return is_n

# Input:  rgb-> vec->ti.Vector([{3,float}])
# Output: hex-> integer looks like 0xFFFFFF
@ti.func
def rgb2hex(rgb: ti.template()):  # r, g, b are normalized
    return ((int(rgb[0] * 255)) << 16) + ((int(rgb[1] * 255)) << 8) + (int(rgb[2] * 255))

# Input:  hex-> integer looks like 0xFFFFFF
# Output: rgb-> vec->ti.Vector([{3,float}])
@ti.func
def hex2rgb(hex: int):  # r, g, b are normalized
    return float(ti.Vector([(hex & 0xFF0000) >> 16, (hex & 0x00FF00) >> 8, (hex & 0x0000FF)])) / 255

# Input:  path-> String
# Output: verts_array-> numpy array (num*dim) dtype=float32  
def read_ply(path):
    obj_ply = PlyData.read(path)
    obj_verts = obj_ply['vertex'].data
    verts_array = np.array([[x, y, z] for x, y, z in obj_verts])
    return verts_array

def write_ply(path, frame_num, dim, num, pos):
    if dim == 3:
        list_pos = [(pos[i, 0], pos[i, 1], pos[i, 2]) for i in range(num)]
    elif dim == 2:
        list_pos = [(pos[i, 0], pos[i, 1], 0) for i in range(num)]
    else:
        print('write_ply(): dim exceeds default values')
        return
    np_pos = np.array(list_pos, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el_pos = PlyElement.describe(np_pos, 'vertex')
    PlyData([el_pos]).write(str(path) + '_' + str(frame_num) + '.ply')


############################################### GUI funcs ###############################################
@ti.kernel
def to_gui_pos(obj: ti.template(), config: ti.template()):
    for i in range(obj.part_num[None]):
        obj.gui_2d_pos[i] = (obj.pos[i] - config.sim_space_lb[None]) / (config.sim_space_rt[None] - config.sim_space_lb[None])


def to_gui_radii(relaxing_factor, config):
    return config.part_size[1] / (config.sim_space_rt[None][0] - config.sim_space_lb[None][0]) * config.gui_res[None][0] / 2000 * relaxing_factor


def to_gui_color(obj):
    return obj.color.to_numpy()[:obj.part_num[None]]


def set_unused_par(obj, config):
    # temporary method: throw the unused particles away so they aren't rendered
    unused_pos = ti.Vector.field(config.dim[None], float, ())
    unused_pos.from_numpy(np.array([533799.0] * config.dim[None], dtype=np.float32))
    sub_set_unused_par(obj, unused_pos)

@ti.kernel
def sub_set_unused_par(obj: ti.template(), unused_pos: ti.template()):
    for i in range(obj.part_num[None], obj.max_part_num):
        obj.pos[i] = unused_pos[None]
        obj.gui_2d_pos[i] = unused_pos[None]