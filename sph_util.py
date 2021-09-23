from sph_config import *

obj_list = []
tmp_int = ti.field(float, 32)
tmp_val = ti.field(float, 32)
dt = ti.field(float, ())
tmp_val_dim = ti.Vector.field(dim, float, 32)
phase_rest_density = ti.Vector.field(phase_num, float, ())
phase_rgb = ti.Vector.field(3, float, phase_num)
sim_space_lb = ti.Vector.field(dim, float, ())
sim_space_rt = ti.Vector.field(dim, float, ())
part_size = ti.field(float, 5)
sph_h = ti.field(float, 5)
sph_sig = ti.field(float, 4)
gravity = ti.Vector.field(dim, float, ())
node_dim = ti.Vector.field(dim, float, ())
node_dim_coder = ti.Vector.field(dim, int, ())
neighb_template = ti.Vector.field(dim, int, (neighb_range*2+1)**dim)
fbm_diffusion_term = ti.field(float, ())
fbm_convection_term = ti.field(float, ())

@ti.func
def C(r):
    q = r/sph_h[1]
    tmp = 0.0
    if q <= 0.5:
        tmp = (2*(1-q)**3*q**3-1/64)
    elif q > 0.5 and q < 1:
        tmp = (1-q)**3*q**3
    tmp *= 32/ math.pi/ np_sph_h[1]**3
    return tmp

@ti.func
def W(r):
    q = r/sph_h[1]
    tmp = 0.0
    if q <= 0.5:
        tmp = 6*(q**3-q**2)+1
    elif q > 0.5 and q < 1:
        tmp = 2*(1-q)**3
    tmp *= sph_sig[dim]
    return tmp


@ti.func
def W_grad(r):
    q = r/sph_h[1]
    tmp = 0.0
    if q <= 0.5:
        tmp = 6*(3*q**2-2*q)
    elif q > 0.5 and q < 1:
        tmp = -6*(1-q)**2
    tmp *= sph_sig[dim]/sph_h[1]
    return tmp


@ti.func
def W_lap(x_ij: ti.template(), r, V_j, A: ti.template()):
    return 2*(2+dim)*V_j*W_grad(r)*x_ij.normalized()*A.dot(x_ij)/(0.01*sph_h[2]+r**2)

@ti.func
def rgb2hex(r: float, g: float, b: float): # r, g, b are normalized
    return ((int(r*255))<<16) + ((int(g*255))<<8) + (int(b*255))

@ti.func
def hex2rgb(hex: ti.template()): # r, g, b are normalized
    return float(ti.Vector([(hex&0xFF0000)>>16,(hex&0x00FF00)>>8,(hex&0x0000FF)]))/255

@ti.kernel
def assign_phase_color(hex: int, phase_num: int):
    phase_rgb[phase_num] = hex2rgb(hex)    

@ti.func
def has_negative(vec: ti.template()):
    is_n = False
    for i in ti.static(range(vec.n)):
        if vec[i] < 0:
            is_n = True
    return is_n


@ti.func
def dim_encode(dim: ti.template()):
    return node_dim_coder[None].dot(dim)

# GUI funcs
def to_gui_res(res_0):
    return (res_0, int((np_sim_space_rt[1]-np_sim_space_lb[1])/(np_sim_space_rt[0]-np_sim_space_lb[0])*res_0))
def to_gui_pos(obj):
    return (obj.pos.to_numpy()[:obj.part_num[None]] - np_sim_space_lb) / (np_sim_space_rt-np_sim_space_lb)
def to_gui_multi_radii(obj):
    return part_size[1]*(obj.mass.to_numpy()[:obj.part_num[None]]*0+1)/(np_sim_space_rt[0]-np_sim_space_lb[0])*gui_res_0/2
def to_gui_radii(relaxing_factor=1):
    return part_size[1]/(np_sim_space_rt[0]-np_sim_space_lb[0])*gui_res_0/2*relaxing_factor
def to_gui_color(obj):
    return obj.color.to_numpy()[:obj.part_num[None]]


