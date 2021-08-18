from sph_config import *

obj_list = []
tmp_int = ti.field(float, 32)
tmp_val = ti.field(float, 32)
dt = ti.field(float, ())
tmp_val_dim = ti.Vector.field(dim, float, 32)
phase_rest_density = ti.Vector.field(phase_num, float, ())
sim_space_lb = ti.Vector.field(dim, float, ())
sim_space_rt = ti.Vector.field(dim, float, ())
part_size = ti.field(float, 5)
sph_h = ti.field(float, 5)
sph_sig = ti.field(float, 4)
gravity = ti.Vector.field(dim, float, ())
node_dim = ti.Vector.field(dim, float, ())
node_dim_coder = ti.Vector.field(dim, int, ())
neighb_template = ti.Vector.field(dim, int, (neighb_range*2+1)**dim)

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
def set_value(tar: ti.template(), index, val):
    tar[index] = val

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


