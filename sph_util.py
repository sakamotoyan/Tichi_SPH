from sph_config import *


@ti.func
def C(r):
    q = r / config.kernel_h[1]
    tmp = 0.0
    if q <= 0.5:
        tmp = (2 * (1 - q) ** 3 * q ** 3 - 1 / 64)
    elif q > 0.5 and q < 1:
        tmp = (1 - q) ** 3 * q ** 3
    tmp *= 32 / math.pi / config.kernel_h[1] ** 3
    return tmp


@ti.func
def W(r):
    q = r / config.kernel_h[1]
    tmp = 0.0
    if q <= 0.5:
        tmp = 6 * (q ** 3 - q ** 2) + 1
    elif q > 0.5 and q < 1:
        tmp = 2 * (1 - q) ** 3
    tmp *= config.kernel_sig[config.dim[None]]
    return tmp


@ti.func
def W_grad(r):
    q = r / config.kernel_h[1]
    tmp = 0.0
    if q <= 0.5:
        tmp = 6 * (3 * q ** 2 - 2 * q)
    elif q > 0.5 and q < 1:
        tmp = -6 * (1 - q) ** 2
    tmp *= config.kernel_sig[config.dim[None]] / config.kernel_h[1]
    return tmp


@ti.func
def W_lap(x_ij: ti.template(), r, V_j, A: ti.template()):
    return 2 * (2 + config.dim[None]) * V_j * W_grad(r) * x_ij.normalized() * A.dot(x_ij) / (
                0.01 * config.kernel_h[2] + r ** 2)


@ti.func
def rgb2hex(r: float, g: float, b: float):  # r, g, b are normalized
    return ((int(r * 255)) << 16) + ((int(g * 255)) << 8) + (int(b * 255))


@ti.func
def hex2rgb(hex: ti.template()):  # r, g, b are normalized
    return float(ti.Vector([(hex & 0xFF0000) >> 16, (hex & 0x00FF00) >> 8, (hex & 0x0000FF)])) / 255


@ti.func
def has_negative(vec: ti.template()):
    is_n = False
    for i in ti.static(range(vec.n)):
        if vec[i] < 0:
            is_n = True
    return is_n


@ti.func
def node_encode(pos: ti.template()):
    return int((pos - config.sim_space_lb[None]) // config.kernel_h[1])


@ti.func
def dim_encode(dim: ti.template()):
    return config.neighb_grid_coder[None].dot(dim)


############################################### PLY funcs ###############################################
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
    PlyData([el_pos], text=True).write(str(path) + '_' + str(frame_num) + '.ply')  # text=True ASCII-format PLY file




############################################### GUI funcs ###############################################
@ti.kernel
def to_gui_pos(obj: ti.template()):
    for i in range(obj.part_num[None]):
        obj.gui_2d_pos[i] = (obj.pos[i] - config.sim_space_lb[None]) / (config.sim_space_rt[None] - config.sim_space_lb[None])


def to_gui_radii(relaxing_factor: ti.template()):
    return config.part_size[1] / (config.sim_space_rt[None][0] - config.sim_space_lb[None][0]) * config.gui_res[None][0] / 2000 * relaxing_factor


def to_gui_color(obj):
    return obj.color.to_numpy()[:obj.part_num[None]]

@ti.kernel
def update_color_vector(obj: ti.template()):
    for i in range(obj.part_num[None]):
        color = hex2rgb(obj.color[i])
        obj.color_vector[i] = color


unused_pos = ti.Vector.field(config.dim[None], float, ())
unused_pos.from_numpy(np.array([533799.0] * config.dim[None], dtype=np.float32))
@ti.kernel
def set_unused_par(obj: ti.template()):
    # temporary method: throw the unused particles away so they aren't rendered
    for i in range(obj.part_num[None], obj.max_part_num):
        obj.pos[i] = unused_pos[None]
        obj.gui_2d_pos[i] = unused_pos[None]
