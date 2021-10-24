from sph_config import *

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
def node_encode(pos: ti.template()):
    return int((pos - sim_space_lb[None])//sph_h[1])

@ti.func
def dim_encode(dim: ti.template()):
    return node_dim_coder[None].dot(dim)

""" PLY funcs """
def read_ply(path):
    obj_ply = PlyData.read(path)
    obj_verts = obj_ply['vertex'].data
    verts_array = np.array([[x, y, z] for x,y,z in obj_verts])
    return verts_array

def write_ply(path, frame_num, dim, num, pos):
    if dim==3:
        list_pos = [(pos[i,0], pos[i,1], pos[i,2]) for i in range(num)]
    elif dim==2:
        list_pos = [(pos[i,0], pos[i,1], 0) for i in range(num)]
    else:
        print('write_ply(): dim exceeds default values')
        return
    np_pos = np.array(list_pos, dtype=[('x','f4'),('y','f4'),('z','f4')])
    el_pos = PlyElement.describe(np_pos, 'vertex')
    PlyData([el_pos]).write(str(path)+'_'+str(frame_num)+'.ply')
    
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

def to_gui_pos_np(arr):
    return (arr - np_sim_space_lb) / (np_sim_space_rt-np_sim_space_lb)

@ti.kernel
def update_color_vector(obj: ti.template()):
    for i in range(obj.part_num[None]):
        color = hex2rgb(obj.color[i])
        obj.color_vector[i] = color
