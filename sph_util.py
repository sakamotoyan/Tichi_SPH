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

def np_normalize(v):
    return v/np.linalg.norm(v)

def warn(*message):
    warnings.warn("".join(message), RuntimeWarning)
    if sys.exc_info()[2] is not None:
        traceback.print_exc()

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

def write_full_json(fname, config, obj):
    data = {
        "timeCounter": config.time_counter[None],
        "timeInSimulation": config.time_count[None],
        "timeStep": config.dt[None],
        "fps": config.gui_fps[None],
        "iteration": {
            "divergenceFree_iteration": config.frame_div_iter[None],
            "incompressible_iteration": config.frame_incom_iter[None],
            "sum_iteration": config.frame_div_iter[None] + config.frame_incom_iter[None]
        },
        "energy": {
            "statistics_kinetic_energy": obj.statistics_kinetic_energy[None],
            "statistics_gravity_potential_energy": obj.statistics_gravity_potential_energy[None],
            "sum_energy": obj.statistics_kinetic_energy[None] + obj.statistics_gravity_potential_energy[None]
        }
    }
    s = json.dumps(data)
    with open(fname, "w") as f:
        f.write(s)

def write_files(gui, config, pre_config, obj):
    gui.window.write_image(f"{pre_config.solver_type}\\img\\rf{int(config.gui_fps[None] + 1e-5)}_{config.time_counter[None]}.png")
    write_ply(path=f'{pre_config.solver_type}\\ply\\fluid_pos', frame_num=config.time_counter[None], dim=config.dim[None], num=obj.part_num[None],pos=obj.pos.to_numpy())
    write_full_json(f"{pre_config.solver_type}\\json\\" + "frame" + str(config.time_counter[None]) + ".json", config, obj)
    # numpy.save(f"{solver_type}\\grid_data\\vel_{globalvar.time_counter}", grid.vel.to_numpy())
    np.save(f"{pre_config.solver_type}\\part_data\\vel_{config.time_counter[None]}", obj.vel.to_numpy()[0:obj.part_num[None], :])
    np.save(f"{pre_config.solver_type}\\part_data\\pos_{config.time_counter[None]}", obj.pos.to_numpy()[0:obj.part_num[None], :])

########################################## transformation funcs #########################################
#helper func
def wrap_np_matrix_to_field(matrix):
    f = ti.Matrix.field(matrix.shape[0],matrix.shape[1],float,())
    f.from_numpy(matrix)
    return f

def rotation_matrix(config, *args):
    m = None
    if len(args) == 1 and config.dim[None] == 2: #2d
        a = args[0]
        m = np.array([
            [cos(a), -sin(a), 0],
            [sin(a),  cos(a), 0],
            [0,       0,      1]
        ])
    elif len(args) == 3 and config.dim[None] == 3: #3d
        x, y, z = args
        rx = np.array([
            [1, 0,       0,      0],
            [0, cos(x), -sin(x), 0],
            [0, sin(x),  cos(x), 0],
            [0, 0,       0,      1]
        ])
        ry = np.array([
            [ cos(y), 0, sin(y), 0],
            [ 0,      1, 0,      0],
            [-sin(y), 0, cos(y), 0],
            [ 0,      0, 0,      1]
        ])
        rz = np.array([
            [cos(z), -sin(z), 0, 0],
            [sin(z),  cos(z), 0, 0],
            [0,       0,      1, 0],
            [0,       0,      0, 1]
        ])
        m = np.matmul(rz,np.matmul(ry,rx))
    else:
        raise Exception('transformation ERROR: dimension mismatch')
    return wrap_np_matrix_to_field(m)

def translation_matrix(config, *args):
    dim = config.dim[None]
    if dim != 2 and dim != 3 or dim != len(args):
        raise Exception('transformation ERROR: dimension mismatch')
    m = np.eye(dim + 1)
    for i in range(dim):
        m[i,dim] = args[i]
    return wrap_np_matrix_to_field(m)

def scale_matrix(config, *args):
    dim = config.dim[None]
    if dim != 2 and dim != 3 or dim != len(args):
        raise Exception('transformation ERROR: dimension mismatch')
    m = np.eye(dim + 1)
    for i in range(dim):
        m[i,i] = args[i]
    return wrap_np_matrix_to_field(m)

@ti.func
def transform(pos, transform_matrix):
    n = ti.static(transform_matrix.n)
    tmp = ti.Vector([0.0] * n)
    for j in ti.static(range(n-1)):
        tmp[j] = pos[j]
    tmp[n-1] = 1
    res = transform_matrix @ tmp
    for j in ti.static(range(n-1)):
        pos[j] = res[j]
    return pos

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
        obj.pos_disp[i] = unused_pos[None]
        obj.gui_2d_pos[i] = unused_pos[None]