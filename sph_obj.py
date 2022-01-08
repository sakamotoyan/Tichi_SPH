from taichi.lang.ops import atomic_add, sqrt
from sph_util import *

obj_list = []
dim = sim_dim


@ti.data_oriented
class Fluid:
    def __init__(self, max_part_num):
        self.max_part_num = max_part_num
        self.part_num = ti.field(int, ())
        self.uid = len(obj_list)  # uid of the Fluid object
        obj_list.append(self)

        # utils
        self.ones = ti.field(int)
        self.flag = ti.field(int)  # ? OBSOLETE
        self.general_flag = ti.field(int, ())  # ? OBSOLETE todo
        self.pushed_part_seq = ti.Vector.field(dim, int, ())  # ? OBSOLETE todo
        self.pushed_part_seq_coder = ti.field(int, dim)  # ? OBSOLETE todo

        # Physical properties of particles
        self.color = ti.field(int)
        self.color_vector = ti.Vector.field(3, float)  # for ggui to show
        self.mass = ti.field(float)
        self.rest_density = ti.field(float)
        self.rest_volume = ti.field(float)
        self.pressure = ti.field(float)
        self.pressure_force = ti.Vector.field(dim, float)
        self.volume_frac = ti.Vector.field(phase_num, float)
        self.volume_frac_tmp = ti.Vector.field(phase_num, float)
        self.pos = ti.Vector.field(dim, float)  # position
        self.gui_2d_pos = ti.Vector.field(dim, float)  # for ggui to show
        self.vel = ti.Vector.field(dim, float)  # velocity
        self.vel_adv = ti.Vector.field(dim, float)
        self.acce = ti.Vector.field(dim, float)  # acceleration
        self.acce_adv = ti.Vector.field(dim, float)

        # energy
        self.statistics_kinetic_energy = ti.field(float, ())  # total kinetic energy of particles
        self.statistics_gravity_potential_energy = ti.field(float, ())  # total gravitational potential energy of particles

        # for slover
        self.W = ti.field(float)
        self.W_grad = ti.Vector.field(dim, float)
        self.compression = ti.field(float, ())  # compression rate gamma for [VFSPH]
        self.sph_compression = ti.field(float)  # diff from compression?
        self.sph_density = ti.field(float)  # density computed from sph approximation
        self.psi_adv = ti.field(float)
        self.alpha = ti.field(float)  # alpha for [DFSPH] and [VFSPH]
        self.alpha_1 = ti.Vector.field(dim, float)  # 1st term of alpha
        self.alpha_2 = ti.field(float)  # 2nd term of alpha
        self.drift_vel = ti.Vector.field(dim, float)
        # FBM
        self.fbm_zeta = ti.field(float)
        self.fbm_acce = ti.static(self.acce)
        self.normal = ti.Vector.field(dim, float)  # surface normal in [AKINCI12] for computing curvature force in surface tension

        # neighb
        self.neighb_cell_seq = ti.field(int)  # the seq of the grid which particle is located
        self.neighb_in_cell_seq = ti.field(int)  # the seq of the particle in the grid
        self.neighb_cell_structured_seq = ti.Vector.field(dim, int)  # the structured seq of the grid

        # [VFSPH] and [DFSPH] use the same framework, aliases for interchangeable variables
        if solver_type == 'VFSPH':
            self.X = ti.static(self.rest_volume)
            self.sph_psi = ti.static(self.sph_compression)
            self.rest_psi = ti.static(self.ones)
        elif solver_type == 'DFSPH':
            self.X = ti.static(self.mass)
            self.sph_psi = ti.static(self.sph_density)
            self.rest_psi = ti.static(self.rest_density)

        # put for-each-particle attributes in this list to register them!
        self.attr_list = [self.color, self.color_vector, self.mass, self.rest_density, self.rest_volume, self.pressure,self.pressure_force,
                          self.volume_frac, self.volume_frac_tmp, self.pos, self.gui_2d_pos, self.vel, self.vel_adv,self.acce, self.acce_adv,
                          self.W, self.W_grad, self.sph_density, self.sph_compression, self.psi_adv, self.alpha, self.alpha_1, self.alpha_2, self.fbm_zeta, self.normal,
                          self.neighb_cell_seq, self.neighb_in_cell_seq, self.neighb_cell_structured_seq, self.ones,self.flag]

        # allocate memory for attributes (1-D fields)
        for attr in self.attr_list:
            ti.root.dense(ti.i, self.max_part_num).place(attr)  # SOA(see Taichi advanced layout: https://docs.taichi.graphics/docs/lang/articles/advanced/layout#from-shape-to-tirootx)
        # allocate memory for drift velocity (2-D field)
        ti.root.dense(ti.i, self.max_part_num).dense(ti.j, phase_num).place(self.drift_vel)
        self.attr_list.append(self.drift_vel)  # add drift velocity to attr_list

        self.init()

    def init(self):
        for attr in self.attr_list:
            attr.fill(0)
        self.ones.fill(1)

    # update mass for volume fraction multiphase
    @ti.kernel
    def update_mass(self):
        for i in range(self.part_num[None]):
            self.mass[i] = config.phase_rest_density[None].dot(self.volume_frac[i])

    # helper function for scene_add functions
    def scene_add_help_centering(self, start_pos, end_pos, spacing):
        end_pos = np.array(end_pos, dtype=np.float32)
        start_pos = np.array(start_pos, dtype=np.float32)
        matrix_shape = ((end_pos - start_pos + 1e-7) / spacing).astype(np.int32)
        padding = (end_pos - start_pos - matrix_shape * spacing) / 2
        return matrix_shape, padding

    # add n dimension cube to scene
    def scene_add_cube(self, start_pos, end_pos, volume_frac, vel, color,
                       relaxing_factor):  # add relaxing factor for each cube
        spacing = config.part_size[1] * relaxing_factor
        matrix_shape, padding = self.scene_add_help_centering(start_pos, end_pos, spacing)
        self.push_matrix(np.ones(matrix_shape, dtype=np.bool_), start_pos + padding, spacing, volume_frac, vel, color)

    #add particles from inlet
    def scene_add_from_inlet(self, center, size, norm, speed, volume_frac, color,
                       relaxing_factor):
        spacing = config.part_size[1] * relaxing_factor
        matrix_shape, padding = self.scene_add_help_centering([0]*len(size), size, spacing)
        seq=[]
        if len(matrix_shape) == 2:
            if norm[0]==0:
                u=np.array([1,0,0])
            else:
                u=np_normalize(np.array([-norm[2]/norm[0],0,1]))
            v=np_normalize(np.cross(u,norm))
            size=np.array(size)-padding
            start=np.array(center)-(size[0]*u+size[1]*v)/2
            for i in range(matrix_shape[0]):
                for j in range(matrix_shape[1]):
                    seq.append(start+(float(i)*u+float(j)*v)*spacing)
        else:
            raise Exception('scenario ERROR: can only add 3D inlets.')
        pos_seq=np.array(seq)
        vel=np.array(norm)*speed
        self.push_part_seq(len(pos_seq), pos_seq, ti.Vector(volume_frac), ti.Vector(vel), color)

    # add 3D or 2D hollow box to scene, with several layers
    def scene_add_box(self, start_pos, end_pos, layers, volume_frac, vel, color, relaxing_factor):
        spacing = config.part_size[1] * relaxing_factor
        matrix_shape, padding = self.scene_add_help_centering(start_pos, end_pos, spacing)
        box = np.ones(matrix_shape, dtype=np.bool_)
        if len(matrix_shape) == 2:
            box[layers: matrix_shape[0] - layers, layers: matrix_shape[1] - layers] = False
        elif len(matrix_shape) == 3:
            box[layers: matrix_shape[0] - layers, layers: matrix_shape[1] - layers,
            layers: matrix_shape[2] - layers] = False
        else:
            raise Exception('scenario error: can only add 2D or 3D boxes')
        self.push_matrix(box, start_pos + padding, spacing, volume_frac, vel, color)

    def push_part_from_ply(self, p_sum, pos_seq, volume_frac, vel, color: int):
        self.push_part_seq(p_sum, pos_seq, ti.Vector(volume_frac), ti.Vector(vel), color)

    # add particles according to true and false in the matrix
    # matrix: np array (dimension: dim, dtype: np.bool)
    def push_matrix(self, matrix, start_position, spacing, volume_frac, vel, color):
        if len(matrix.shape) != dim:
            raise Exception('scenario error: wrong object dimension')
        index = np.where(matrix == True)
        pos_seq = np.stack(index, axis=1) * spacing + start_position
        self.push_part_seq(len(pos_seq), pos_seq, ti.Vector(volume_frac), ti.Vector(vel), color)

    @ti.kernel
    def push_part_seq(self, pushed_part_num: int, pos_seq: ti.ext_arr(), volume_frac: ti.template(), vel: ti.template(),
                      color: int):
        current_part_num = self.part_num[None]
        new_part_num = current_part_num + pushed_part_num
        for i in range(pushed_part_num):
            for j in ti.static(range(dim)):
                self.pos[i + current_part_num][j] = pos_seq[i, j]
            self.volume_frac[i + current_part_num] = volume_frac
            self.vel[i + current_part_num] = vel
            self.rest_volume[i + current_part_num] = config.part_size[dim]  # todo 1
            self.color[i + current_part_num] = color
        self.part_num[None] = new_part_num
        for i in range(self.part_num[None]):
            self.rest_density[i] = config.phase_rest_density[None].dot(self.volume_frac[i])  # todo 2
            self.mass[i] = self.rest_density[i] * self.rest_volume[i]

    @ti.kernel
    def push_cube(self, lb: ti.template(), rt: ti.template(), mask: ti.template(), volume_frac: ti.template(),
                  color: int, relaxing_factor: ti.template()):
        current_part_num = self.part_num[None]
        # generate seq (number of particles to push for each dimension)
        self.pushed_part_seq[None] = int(ti.ceil((rt - lb) / config.part_size[1] / relaxing_factor))
        self.pushed_part_seq[None] *= mask
        for i in ti.static(range(dim)):
            if self.pushed_part_seq[None][i] == 0:
                self.pushed_part_seq[None][i] = 1  # at least push one
        # coder for seq
        tmp = 1
        for i in ti.static(range(dim)):
            self.pushed_part_seq_coder[i] = tmp
            tmp *= self.pushed_part_seq[None][i]
        # new part num
        pushed_part_num = 1
        for i in ti.static(range(dim)):
            pushed_part_num *= self.pushed_part_seq[None][i]
        new_part_num = current_part_num + pushed_part_num
        # inject pos [1/2]
        for i in range(pushed_part_num):
            tmp = i
            for j in ti.static(range(dim - 1, -1, -1)):
                self.pos[i + current_part_num][j] = tmp // self.pushed_part_seq_coder[j]
                tmp = tmp % self.pushed_part_seq_coder[j]
        # inject pos [2/2]
        # pos seq times part size minus lb
        for i in range(pushed_part_num):
            self.pos[i + current_part_num] *= config.part_size[1] * relaxing_factor
            self.pos[i + current_part_num] += lb
        # inject volume_frac & rest_volume & color
        for i in range(pushed_part_num):
            self.volume_frac[i + current_part_num] = volume_frac
            self.rest_volume[i + current_part_num] = config.part_size[dim]
            self.color[i + current_part_num] = color
        # update part num
        self.part_num[None] = new_part_num
        # update mass and rest_density
        for i in range(self.part_num[None]):
            self.rest_density[i] = config.phase_rest_density[None].dot(self.volume_frac[i])
            self.mass[i] = self.rest_density[i] * self.rest_volume[i]

    def inc_unit(self, seq, length, lim, cur_dim):
        for i in range(length):
            if not seq[cur_dim][i] < lim[cur_dim]:
                seq[cur_dim + 1][i] = seq[cur_dim][i] // lim[cur_dim]
                seq[cur_dim][i] = seq[cur_dim][i] % lim[cur_dim]

    def push_2d_cube(self, center_pos, size, volume_frac, color: int, relaxing_factor, layer=0):
        lb = -np.array(size) / 2 + np.array(center_pos)
        rt = np.array(size) / 2 + np.array(center_pos)
        mask = np.ones(dim, np.int32)
        if layer == 0:
            self.push_cube(ti.Vector(lb), ti.Vector(rt), ti.Vector(mask), ti.Vector(volume_frac), color)
        elif layer > 0:
            cube_part = np.zeros(dim, np.int32)
            cube_part[:] = np.ceil(np.array(size) / config.part_size[1] / relaxing_factor)[:]
            for i in range(cube_part.shape[0]):
                if cube_part[i] < layer * 2:
                    layer = int(np.floor(cube_part[i] / 2))
            sum = int(1)
            for i in range(cube_part.shape[0]):
                sum *= cube_part[i]
            np_pos_seq = np.zeros(shape=(dim + 1, sum), dtype=np.int32)
            counter = int(0)
            for i in range(sum):
                np_pos_seq[0][i] = counter
                counter += 1
            for i in range(0, dim - 1):
                self.inc_unit(np_pos_seq, sum, cube_part, i)
            p_sum = int(0)
            for i in range(layer):
                for j in range(dim):
                    for k in range(sum):
                        if (np_pos_seq[j][k] == (0 + i) or np_pos_seq[j][k] == (cube_part[j] - i - 1)) and \
                                np_pos_seq[dim][k] == 0:
                            np_pos_seq[dim][k] = 1
                            p_sum += 1
            pos_seq = np.zeros((p_sum, dim), np.float32)
            counter = int(0)
            for i in range(sum):
                if np_pos_seq[dim][i] > 0:
                    pos_seq[counter][:] = np_pos_seq[0:dim, i]
                    counter += 1
            pos_seq *= config.part_size[1] * relaxing_factor
            pos_seq -= (np.array(center_pos) + np.array(size) / 2)
            self.push_part_seq(p_sum, pos_seq, ti.Vector(volume_frac), color)


class Part_buffer:
    def __init__(self, part_num):
        self.rest_volume = np.zeros(shape=part_num, dtype=np.float32)
        self.volume_frac = np.zeros(shape=(phase_num, part_num), dtype=np.float32)
        self.pos = np.zeros(shape=(dim, part_num), dtype=np.float32)


max_part_num = config.fluid_max_part_num[None] + config.bound_max_part_num[None]
@ti.data_oriented
class Ngrid:
    def __init__(self):
        self.node_part_count = ti.field(int)
        self.node_part_shift = ti.field(int)
        self.node_part_shift_count = ti.field(int)
        self.part_pid_in_node = ti.field(int)
        self.part_uid_in_node = ti.field(int)

        ti.root.dense(ti.i, config.node_num[None]).place(self.node_part_count)
        ti.root.dense(ti.i, config.node_num[None]).place(self.node_part_shift)
        ti.root.dense(ti.i, config.node_num[None]).place(self.node_part_shift_count)
        ti.root.dense(ti.i, max_part_num).place(self.part_pid_in_node)
        ti.root.dense(ti.i, max_part_num).place(self.part_uid_in_node)

    @ti.kernel
    def clear_node(self):
        for i in range(config.node_num[None]):
            self.node_part_count[i] = 0

    @ti.kernel
    def encode(self, obj: ti.template()):
        for i in range(obj.part_num[None]):
            obj.neighb_cell_structured_seq[i] = node_encode(obj.pos[i])
            obj.neighb_cell_seq[i] = dim_encode(obj.neighb_cell_structured_seq[i])
            if 0 < obj.neighb_cell_seq[i] < config.node_num[None]:
                ti.atomic_add(self.node_part_count[obj.neighb_cell_seq[i]], 1)

    @ti.kernel
    def mem_shift(self):
        sum = ti.Vector([0])
        for i in range(config.node_num[None]):
            self.node_part_shift[i] = ti.atomic_add(
                sum[0], self.node_part_count[i])
            self.node_part_shift_count[i] = self.node_part_shift[i]

    @ti.kernel
    def fill_node(self, obj: ti.template()):
        for i in range(obj.part_num[None]):
            if 0 < obj.neighb_cell_seq[i] < config.node_num[None]:
                obj.neighb_in_cell_seq[i] = atomic_add(
                    self.node_part_shift_count[obj.neighb_cell_seq[i]], 1)
                self.part_pid_in_node[obj.neighb_in_cell_seq[i]] = i
                self.part_uid_in_node[obj.neighb_in_cell_seq[i]] = obj.uid


shape = tuple((config.sim_space_rt[None].to_numpy() - config.sim_space_lb[None].to_numpy() / config.part_size[1] * config.neighb_grid_size_TO_global_part_size[None]).astype(np.int32))
# for particle-grid mapping
@ti.data_oriented
class Grid:
    def __init__(self):
        self.shape = shape  # number of grids on each dimension
        self.lb = config.sim_space_lb[None].to_numpy()  # smallest coordination of the grid
        self.dist = config.part_size[1] * config.neighb_grid_size_TO_global_part_size[None]  # distance between each grid cell
        self.size = 1
        for i in range(len(shape)):
            self.size *= shape[i]
        self.vel = ti.Vector.field(dim, float, shape=self.shape)
        self.pos = ti.Vector.field(dim, float, shape=self.shape)
        self.init_pos()

    @ti.kernel
    def init_pos(self):
        for I in ti.grouped(self.pos):
            self.pos[I] = config.sim_space_lb[None] + I * self.dist


class GlobalVariable:
    def __init__(self):
        self.is_first_time = True
        self.time_real = 0
        self.time_start = 0

        self.time_count = float(0)
        self.time_counter = int(0)
        self.step_counter = int(0)

        self.frame_div_iter = 0
        self.frame_incom_iter = 0
        self.div_iter_count = 0
        self.incom_iter_count = 0

        # self.show_window = False
        self.show_window = True
        self.show_bound = False
        self.show_help = True
        self.show_run_info = True
        self.op_system_run = False
        self.op_write_file = False