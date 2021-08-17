from taichi.lang.ops import atomic_add
from sph_util import *


@ti.data_oriented
class Fluid:
    def __init__(self, max_part_num):
        self.max_part_num = max_part_num
        self.part_num = ti.field(ti.i32, ())
        self.uid = len(obj_list)
        self.pushed_part_seq = ti.Vector.field(dim, ti.i32, ())
        self.pushed_part_seq_coder = ti.field(ti.i32, dim)
        obj_list.append(self)
        self.compression = ti.field(ti.f32, ())

        self.node_code = ti.field(ti.i32)
        self.node_code_seq = ti.field(ti.i32)
        self.node = ti.Vector.field(dim, ti.i32)
        self.ones = ti.field(ti.i32)
        self.color = ti.field(ti.i32)
        self.W = ti.field(ti.f32)
        self.W_grad = ti.Vector.field(dim, ti.f32)
        self.volume_frac = ti.Vector.field(phase_num, ti.f32)
        self.mass = ti.field(ti.f32)
        self.rest_density = ti.field(ti.f32)
        self.rest_volume = ti.field(ti.f32)
        self.sph_compression = ti.field(ti.f32)
        self.sph_density = ti.field(ti.f32)
        self.psi_adv = ti.field(ti.f32)
        self.pressure = ti.field(ti.f32)
        self.pressure_force = ti.Vector.field(dim, ti.f32)
        self.pos = ti.Vector.field(dim, ti.f32)
        self.vel = ti.Vector.field(dim, ti.f32)
        self.vel_adv = ti.Vector.field(dim, ti.f32)
        self.acce = ti.Vector.field(dim, ti.f32)
        self.acce_adv = ti.Vector.field(dim, ti.f32)
        self.alpha = ti.field(ti.f32)
        self.alpha_1 = ti.Vector.field(dim, ti.f32)
        self.alpha_2 = ti.field(ti.f32)

        # self.X = ti.static(self.mass)
        # self.sph_psi = ti.static(self.sph_density)
        # self.rest_psi = ti.static(self.rest_density)

        self.X = ti.static(self.rest_volume)
        self.sph_psi = ti.static(self.sph_compression)
        self.rest_psi = ti.static(self.ones)

        self.attr_list = [self.node_code, self.node_code_seq, self.node, self.ones, self.color, self.W, self.W_grad, self.volume_frac, self.mass, self.rest_density, self.rest_volume, self.sph_density,
                          self.sph_compression, self.psi_adv, self.pressure, self.pressure_force, self.pos, self.vel, self.vel_adv, self.acce, self.acce_adv, self.alpha, self.alpha_1, self.alpha_2]

        for attr in self.attr_list:
            ti.root.dense(ti.i, self.max_part_num).place(attr)

    def set_zero(self):
        for attr in self.attr_list:
            attr.fill(0)

    @ti.kernel
    def update_mass(self):
        for i in range(self.part_num[None]):
            self.mass[i] = phase_rest_density[None].dot(self.volume_frac[i])

    @ti.kernel
    def push_cube(self, lb: ti.template(), rt: ti.template(), mask: ti.template(), volume_frac: ti.template(), color:ti.i32, relaxing_factor: ti.f32):
        current_part_num = self.part_num[None]
        # generate seq
        self.pushed_part_seq[None] = int(ti.ceil((rt-lb)/part_size[1]/relaxing_factor))
        self.pushed_part_seq[None] *= mask
        for i in ti.static(range(dim)):
            if self.pushed_part_seq[None][i] == 0:
                self.pushed_part_seq[None][i] = 1
        # coder for seq
        tmp=1
        for i in ti.static(range(dim)):
            self.pushed_part_seq_coder[i] = tmp
            tmp *= self.pushed_part_seq[None][i]
        # new part num
        pushed_part_num = 1
        for i in ti.static(range(dim)):
            pushed_part_num *= self.pushed_part_seq[None][i]
        new_part_num = current_part_num+pushed_part_num
        # inject pos [1/2]
        for i in range(pushed_part_num):
            tmp = i
            for j in ti.static(range(dim-1,-1,-1)):
                self.pos[i+current_part_num][j] = tmp//self.pushed_part_seq_coder[j]
                tmp = tmp % self.pushed_part_seq_coder[j]
        # inject pos [2/2]
        # pos seq times part size minus lb
        for i in range(pushed_part_num):
            self.pos[i+current_part_num] *= part_size[1]*relaxing_factor
            self.pos[i+current_part_num] += lb
        # inject volume_frac & rest_volume & color
        for i in range(pushed_part_num):
            self.volume_frac[i+current_part_num] = volume_frac
            self.rest_volume[i+current_part_num] = part_size[dim]
            self.color[i+current_part_num] = color
        # update part num
        self.part_num[None] = new_part_num
        # update mass and rest_density
        for i in range(self.part_num[None]):
            self.rest_density[i] = phase_rest_density[None].dot(self.volume_frac[i])
            self.mass[i] = self.rest_density[i]*self.rest_volume[i]
    
    @ti.kernel
    def push_part_seq(self, pushed_part_num: ti.i32, pos_seq: ti.ext_arr(), volume_frac: ti.template(), color:ti.i32):
        current_part_num = self.part_num[None]
        new_part_num = current_part_num+pushed_part_num
        for i in range(pushed_part_num):
            for j in ti.static(range(dim)):
                self.pos[i+current_part_num][j] = pos_seq[i,j]
            self.volume_frac[i+current_part_num] = volume_frac
            self.rest_volume[i+current_part_num] = part_size[dim]
            self.color[i+current_part_num] = color
        self.part_num[None] = new_part_num
        for i in range(self.part_num[None]):
            self.rest_density[i] = phase_rest_density[None].dot(self.volume_frac[i])
            self.mass[i] = self.rest_density[i]*self.rest_volume[i]

    def inc_unit(self, seq, length, lim, cur_dim):
        for i in range(length):
            if not seq[cur_dim][i] < lim[cur_dim]:
                seq[cur_dim+1][i] = seq[cur_dim][i] // lim[cur_dim]
                seq[cur_dim][i] = seq[cur_dim][i] % lim[cur_dim]

    def push_2d_cube(self, center_pos, size, volume_frac, color: int, layer=0):
        lb = -np.array(size)/2 + np.array(center_pos)
        rt = np.array(size)/2 + np.array(center_pos)
        mask = np.ones(dim, np.int32)
        relaxing_factor=1.01
        if layer==0:
            self.push_cube(ti.Vector(lb), ti.Vector(rt), ti.Vector(mask), ti.Vector(volume_frac), color, relaxing_factor)
        elif layer>0:
            cube_part = np.zeros(dim, np.int32)
            cube_part[:] = np.ceil(np.array(size) / part_size[1])[:]
            for i in range(cube_part.shape[0]):
                if cube_part[i]<layer*2:
                    layer = int(np.floor(cube_part[i]/2))
            sum = int(1)
            for i in range(cube_part.shape[0]):
                sum *= cube_part[i]
            np_pos_seq = np.zeros(shape=(dim+1, sum),dtype=np.int32)
            counter = int(0)
            for i in range(sum):
                np_pos_seq[0][i] = counter
                counter += 1
            for i in range(0,dim-1):
                self.inc_unit(np_pos_seq, sum, cube_part, i)
            p_sum = int(0)
            for i in range(layer):
                for j in range(dim):
                    for k in range(sum):
                        if (np_pos_seq[j][k] == (0+i) or np_pos_seq[j][k] == (cube_part[j]-i-1)) and np_pos_seq[dim][k]==0:
                            np_pos_seq[dim][k] = 1
                            p_sum += 1
            pos_seq = np.zeros((p_sum,dim), np.float32)
            counter = int(0)
            for i in range(sum):
                if np_pos_seq[dim][i] > 0:
                    pos_seq[counter][:] = np_pos_seq[0:dim,i]
                    counter += 1
            pos_seq *= part_size[1]*relaxing_factor
            pos_seq -= (np.array(center_pos) + np.array(size)/2)
            # print(pos_seq)
            self.push_part_seq(p_sum, pos_seq, ti.Vector(volume_frac), color)
            

            


class Part_buffer:
    def __init__(self, part_num):
        self.rest_volume = np.zeros(shape=part_num, dtype=np.float32)
        self.volume_frac = np.zeros(shape=(phase_num, part_num), dtype=np.float32)
        self.pos = np.zeros(shape=(dim, part_num), dtype=np.float32)


@ti.data_oriented
class Ngrid:
    def __init__(self):
        self.node_part_count = ti.field(ti.i32)
        self.node_part_shift = ti.field(ti.i32)
        self.node_part_shift_count = ti.field(ti.i32)
        self.part_pid_in_node = ti.field(ti.i32)
        self.part_uid_in_node = ti.field(ti.i32)

        ti.root.dense(ti.i, node_num).place(self.node_part_count)
        ti.root.dense(ti.i, node_num).place(self.node_part_shift)
        ti.root.dense(ti.i, node_num).place(self.node_part_shift_count)
        ti.root.dense(ti.i, max_part_num).place(self.part_pid_in_node)
        ti.root.dense(ti.i, max_part_num).place(self.part_uid_in_node)

    @ti.kernel
    def clear_node(self):
        for i in range(node_num):
            self.node_part_count[i] = 0

    @ti.kernel
    def encode(self, obj: ti.template()):
        for i in range(obj.part_num[None]):
            obj.node[i] = int((obj.pos[i] - sim_space_lb[None])//sph_h[1])
            obj.node_code[i] = dim_encode(obj.node[i])
            if 0 < obj.node_code[i] < node_num:
                ti.atomic_add(self.node_part_count[obj.node_code[i]], 1)

    @ti.kernel
    def mem_shift(self):
        sum = ti.Vector([0])
        for i in range(node_num):
            self.node_part_shift[i] = ti.atomic_add(
                sum[0], self.node_part_count[i])
            self.node_part_shift_count[i] = self.node_part_shift[i]

    @ti.kernel
    def fill_node(self, obj: ti.template()):
        for i in range(obj.part_num[None]):
            if 0 < obj.node_code[i] < node_num:
                obj.node_code_seq[i] = atomic_add(
                    self.node_part_shift_count[obj.node_code[i]], 1)
                self.part_pid_in_node[obj.node_code_seq[i]] = i
                self.part_uid_in_node[obj.node_code_seq[i]] = obj.uid
