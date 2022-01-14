from taichi.lang.ops import atomic_add, sqrt
from sph_util import *

obj_list = []

@ti.data_oriented
class Fluid:
    def __init__(self, max_part_num, pre_config, config):

        obj_list.append(self)

        self.max_part_num = max_part_num
        self.part_num = ti.field(int, ())
        self.uid = len(obj_list)  # uid of the Fluid object
        # utils
        self.ones = ti.field(int)
        self.flag = ti.field(int)  # ? OBSOLETE
        self.general_flag = ti.field(int, ())  # ? OBSOLETE todo
        self.pushed_part_seq = ti.Vector.field(config.dim[None], int, ())  # ? OBSOLETE todo
        self.pushed_part_seq_coder = ti.field(int, config.dim[None])  # ? OBSOLETE todo

        # Physical properties of particles
        self.color = ti.field(int)
        self.color_vector = ti.Vector.field(3, float)  # for ggui to show
        self.mass = ti.field(float)
        self.rest_density = ti.field(float)
        self.rest_volume = ti.field(float)
        self.pressure = ti.field(float)
        self.pressure_force = ti.Vector.field(config.dim[None], float)
        self.volume_frac = ti.Vector.field(config.phase_num[None], float)
        self.volume_frac_tmp = ti.Vector.field(config.phase_num[None], float)
        self.pos = ti.Vector.field(config.dim[None], float)  # position
        self.gui_2d_pos = ti.Vector.field(config.dim[None], float)  # for ggui to show
        self.vel = ti.Vector.field(config.dim[None], float)  # velocity
        self.vel_adv = ti.Vector.field(config.dim[None], float)
        self.acce = ti.Vector.field(config.dim[None], float)  # acceleration
        self.acce_adv = ti.Vector.field(config.dim[None], float)

        # energy
        self.statistics_kinetic_energy = ti.field(float, ())  # total kinetic energy of particles
        self.statistics_gravity_potential_energy = ti.field(float, ())  # total gravitational potential energy of particles

        # for slover
        self.W = ti.field(float)
        self.W_grad = ti.Vector.field(config.dim[None], float)
        self.compression = ti.field(float, ())  # compression rate gamma for [VFSPH]
        self.sph_compression = ti.field(float)  # diff from compression?
        self.sph_density = ti.field(float)  # density computed from sph approximation
        self.psi_adv = ti.field(float)
        self.alpha = ti.field(float)  # alpha for [DFSPH] and [VFSPH]
        self.alpha_1 = ti.Vector.field(config.dim[None], float)  # 1st term of alpha
        self.alpha_2 = ti.field(float)  # 2nd term of alpha
        self.drift_vel = ti.Vector.field(config.dim[None], float)
        # FBM
        self.fbm_zeta = ti.field(float)
        self.fbm_acce = ti.static(self.acce)
        self.normal = ti.Vector.field(config.dim[None], float)  # surface normal in [AKINCI12] for computing curvature force in surface tension

        # neighb
        self.neighb_cell_seq = ti.field(int)  # the seq of the grid which particle is located
        self.neighb_in_cell_seq = ti.field(int)  # the seq of the particle in the grid
        self.neighb_cell_structured_seq = ti.Vector.field(config.dim[None], int)  # the structured seq of the grid

        # [VFSPH] and [DFSPH] use the same framework, aliases for interchangeable variables
        if pre_config.solver_type == 'VFSPH':
            self.X = ti.static(self.rest_volume)
            self.sph_psi = ti.static(self.sph_compression)
            self.rest_psi = ti.static(self.ones)
        elif pre_config.solver_type == 'DFSPH':
            self.X = ti.static(self.mass)
            self.sph_psi = ti.static(self.sph_density)
            self.rest_psi = ti.static(self.rest_density)

        # display
        self.pos_disp = ti.Vector.field(config.dim[None], float)

        # wcsph_21
        self.F_mid = ti.Vector.field(config.dim[None], float)
        self.vel_mid_phase = ti.Vector.field(config.dim[None], float)
        self.vel_mid = ti.Vector.field(config.dim[None], float)
        self.vel_phase = ti.Vector.field(config.dim[None], float)
        self.lamb = ti.field(float)

        # put for-each-particle attributes in this list to register them!
        self.attr_list = [self.color, self.color_vector, self.mass, self.rest_density, self.rest_volume, self.pressure,self.pressure_force,
                          self.volume_frac, self.volume_frac_tmp, self.pos, self.gui_2d_pos, self.vel, self.vel_adv,self.acce, self.acce_adv,
                          self.W, self.W_grad, self.sph_density, self.sph_compression, self.psi_adv, self.alpha, self.alpha_1, self.alpha_2, self.fbm_zeta, self.normal,
                          self.neighb_cell_seq, self.neighb_in_cell_seq, self.neighb_cell_structured_seq, self.ones,self.flag, self.pos_disp,
                          self.F_mid, self.vel_mid, self.lamb]

        # allocate memory for attributes (1-D fields)
        for attr in self.attr_list:
            ti.root.dense(ti.i, self.max_part_num).place(attr)  # SOA(see Taichi advanced layout: https://docs.taichi.graphics/docs/lang/articles/advanced/layout#from-shape-to-tirootx)
        # allocate memory for drift velocity (2-D field)
        ti.root.dense(ti.i, self.max_part_num).dense(ti.j, config.phase_num[None]).place(self.drift_vel)
        self.attr_list.append(self.drift_vel)  # add drift velocity to attr_list
        ti.root.dense(ti.i, self.max_part_num).dense(ti.j, config.phase_num[None]).place(self.vel_mid_phase)
        self.attr_list.append(self.vel_mid_phase)  # add drift velocity to attr_list
        ti.root.dense(ti.i, self.max_part_num).dense(ti.j, config.phase_num[None]).place(self.vel_phase)
        self.attr_list.append(self.vel_phase)  # add drift velocity to attr_list

        self.obj_part_range_from_name={}

        self.unused_pos = ti.Vector.field(config.dim[None], float, ())
        self.unused_pos.from_numpy(np.array([533799.0] * config.dim[None], dtype=np.float32))

        self.init()

    def init(self):
        for attr in self.attr_list:
            attr.fill(0)
        self.ones.fill(1)

    # update mass for volume fraction multiphase
    @ti.kernel
    def update_mass(self, config: ti.template()):
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
                       relaxing_factor, config):  # add relaxing factor for each cube
        spacing = config.part_size[1] * relaxing_factor
        matrix_shape, padding = self.scene_add_help_centering(start_pos, end_pos, spacing)

        self.push_matrix(np.ones(matrix_shape, dtype=np.bool_), start_pos + padding, spacing, volume_frac, vel, color, config)

    #add particles from inlet
    def scene_add_from_inlet(self, center, size, norm, speed, volume_frac, color,
                       relaxing_factor, config):
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
        self.push_part_seq(len(pos_seq), color, pos_seq, ti.Vector(volume_frac), ti.Vector(vel), config)


    # add 3D or 2D hollow box to scene, with several layers
    def scene_add_box(self, start_pos, end_pos, layers, volume_frac, vel, color, relaxing_factor, config):
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
        self.push_matrix(box, start_pos + padding, spacing, volume_frac, vel, color, config)

    def scene_add_ply(self, p_sum, pos_seq, volume_frac, vel, color, config):
        self.push_part_seq(p_sum, color, pos_seq, ti.Vector(volume_frac), ti.Vector(vel), config)

    # add particles according to true and false in the matrix
    # matrix: np array (dimension: dim, dtype: np.bool)
    def push_matrix(self, matrix, start_position, spacing, volume_frac, vel, color, config):
        if len(matrix.shape) != config.dim[None]:
            raise Exception('push_matrix() [scenario error]: wrong object dimension')
        index = np.where(matrix == True)
        pos_seq = np.stack(index, axis=1) * spacing + start_position
        self.push_part_seq(len(pos_seq), color, pos_seq, ti.Vector(volume_frac), ti.Vector(vel), config)


    @ti.kernel
    def push_pos_seq(self, pos_seq: ti.template(),pushed_part_num: int, current_part_num: int, config: ti.template()):
        dim = ti.static(config.gravity.n)
        for i in range(pushed_part_num):
            i_p = i + current_part_num
            for j in ti.static(range(dim)):
                self.pos[i_p][j] = pos_seq[i][j]


    @ti.kernel
    def push_attrs_seq(self, color: int, volume_frac: ti.template(), vel: ti.template(), pushed_part_num: int, current_part_num: int, config: ti.template()):
        phase_num = ti.static(config.phase_rest_density.n)
        for i in range(pushed_part_num):
            i_p = i + current_part_num
            self.volume_frac[i_p] = volume_frac
            self.vel[i_p] = vel
            self.rest_volume[i_p] = config.part_size[config.dim[None]]  # todo 1
            self.color[i_p] = color
            self.color_vector[i_p] = hex2rgb(color)
            self.rest_density[i_p] = config.phase_rest_density[None].dot(self.volume_frac[i_p])
            self.mass[i_p] = self.rest_density[i_p] * self.rest_volume[i_p]
            for k in ti.static(range(phase_num)):
                self.vel_phase[i_p, k] = vel


    def push_part_seq(self, pushed_part_num, color, pos_seq, volume_frac, vel, config):
        print('push ',pushed_part_num, ' particles')
        current_part_num = self.part_num[None]
        new_part_num = current_part_num + pushed_part_num
        pos_seq_ti = ti.Vector.field(config.dim[None], float, pushed_part_num)
        pos_seq_ti.from_numpy(pos_seq)
        self.push_pos_seq(pos_seq_ti, pushed_part_num, current_part_num, config)
        self.push_attrs_seq(color, volume_frac, vel, pushed_part_num, current_part_num, config)
        self.part_num[None] = new_part_num


    @ti.kernel
    def push_cube(self, lb: ti.template(), rt: ti.template(), mask: ti.template(), volume_frac: ti.template(),
                  color: int, relaxing_factor: ti.template(), config: ti.template()):
        current_part_num = self.part_num[None]
        # generate seq (number of particles to push for each dimension)
        self.pushed_part_seq[None] = int(ti.ceil((rt - lb) / config.part_size[1] / relaxing_factor))
        self.pushed_part_seq[None] *= mask
        dim = ti.static(config.gravity.n)
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
            self.rest_volume[i + current_part_num] = config.part_size[config.dim[None]]
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

    def push_2d_cube(self, center_pos, size, volume_frac, color: int, relaxing_factor, config, layer=0):
        lb = -np.array(size) / 2 + np.array(center_pos)
        rt = np.array(size) / 2 + np.array(center_pos)
        mask = np.ones(config.dim[None], np.int32)
        if layer == 0:
            self.push_cube(ti.Vector(lb), ti.Vector(rt), ti.Vector(mask), ti.Vector(volume_frac), color)
        elif layer > 0:
            cube_part = np.zeros(config.dim[None], np.int32)
            cube_part[:] = np.ceil(np.array(size) / config.part_size[1] / relaxing_factor)[:]
            for i in range(cube_part.shape[0]):
                if cube_part[i] < layer * 2:
                    layer = int(np.floor(cube_part[i] / 2))
            sum = int(1)
            for i in range(cube_part.shape[0]):
                sum *= cube_part[i]
            np_pos_seq = np.zeros(shape=(config.dim[None] + 1, sum), dtype=np.int32)
            counter = int(0)
            for i in range(sum):
                np_pos_seq[0][i] = counter
                counter += 1
            for i in range(0, config.dim[None] - 1):
                self.inc_unit(np_pos_seq, sum, cube_part, i)
            p_sum = int(0)
            for i in range(layer):
                for j in range(config.dim[None]):
                    for k in range(sum):
                        if (np_pos_seq[j][k] == (0 + i) or np_pos_seq[j][k] == (cube_part[j] - i - 1)) and \
                                np_pos_seq[config.dim[None]][k] == 0:
                            np_pos_seq[config.dim[None]][k] = 1
                            p_sum += 1
            pos_seq = np.zeros((p_sum, config.dim[None]), np.float32)
            counter = int(0)
            for i in range(sum):
                if np_pos_seq[config.dim[None]][i] > 0:
                    pos_seq[counter][:] = np_pos_seq[0:config.dim[None], i]
                    counter += 1
            pos_seq *= config.part_size[1] * relaxing_factor
            pos_seq -= (np.array(center_pos) + np.array(size) / 2)
            self.push_part_seq(p_sum, color, pos_seq, ti.Vector(volume_frac), config)

    def push_scene_obj(self, param, config):
        pre_part_cnt=self.part_num[None]
        if param['type'] == 'cube':
            self.scene_add_cube(param['start_pos'], param['end_pos'], param['volume_frac'], param['vel'], int(param['color'], 16), param['particle_relaxing_factor'],config)
        elif param['type'] == 'box':
            self.scene_add_box(param['start_pos'], param['end_pos'], param['layers'], param['volume_frac'], param['vel'], int(param['color'], 16), param['particle_relaxing_factor'],config)
        elif param['type'] == 'ply':
            verts = read_ply(param['file_name'])
            self.scene_add_ply(len(verts), verts, param['volume_frac'], param['vel'], int(param['color'], 16),config)
        else:
            raise Exception('scenario ERROR: object type unsupported:',
                param['type'] if 'type' in param else 'None')
        part_range=(pre_part_cnt,self.part_num[None])
        if 'name' in param:
            self.obj_part_range_from_name[param['name']] = part_range

    @ti.kernel
    def update_color_vector_from_color(self):
        for i in range(self.part_num[None]):
            color = hex2rgb(self.color[i])
            self.color_vector[i] = color

    @ti.kernel
    def display_all(self):
        for i in range(self.part_num[None]):
            self.pos_disp[i] = self.pos[i]
    
    @ti.kernel
    def display_part_range(self, start_id: int, end_id: int):
        for i in range(self.part_num[None]):
            if i >= start_id and i < end_id:
                self.pos_disp[i] = self.pos[i]
            else:
                self.pos_disp[i] = self.unused_pos[None]

    def get_part_range_from_name(self, name):
        return self.obj_part_range_from_name[name]

    @ti.kernel
    def set_vel_part_range(self, start_id: int, end_id: int, vel:ti.template()):
        for i in range(start_id,end_id):
            self.vel[i] = vel[None]
    
    @ti.kernel
    def update_pos_part_range(self, start_id: int, end_id: int, config:ti.template()):
        for i in range(start_id,end_id):
            self.pos[i] += self.vel[i] * config.dt[None]

    ######################### transform functions (too slow to be used every timestep) ##########################
    @ti.kernel
    def transform_part_range(self, start_id: int, end_id: int, transform_matrix:ti.template()):
        for i in range(start_id,end_id):
            self.pos[i] = transform(self.pos[i],transform_matrix[None])

    # transform with velocity update
    @ti.kernel
    def move_part_range(self, start_id: int, end_id: int, transform_matrix:ti.template(), config:ti.template()):
        for i in range(start_id,end_id):
            pre_pos = self.pos[i]
            self.pos[i] = transform(self.pos[i],transform_matrix[None])
            self.vel[i] = (self.pos[i] - pre_pos) / config.dt[None]

    def move_scene_obj(self, name, transform_matrix, config):
        a,b = self.get_part_range_from_name(name)
        self.move_part_range(a,b,transform_matrix,config)
    

class Part_buffer:
    def __init__(self, part_num, config):
        self.rest_volume = np.zeros(shape=part_num, dtype=np.float32)
        self.volume_frac = np.zeros(shape=(config.phase_num[None], part_num), dtype=np.float32)
        self.pos = np.zeros(shape=(config.dim[None], part_num), dtype=np.float32)

@ti.data_oriented
class Ngrid:
    def __init__(self, config):
        self.node_part_count = ti.field(int)
        self.node_part_shift = ti.field(int)
        self.node_part_shift_count = ti.field(int)
        self.part_pid_in_node = ti.field(int)
        self.part_uid_in_node = ti.field(int)

        ti.root.dense(ti.i, config.node_num[None]).place(self.node_part_count)
        ti.root.dense(ti.i, config.node_num[None]).place(self.node_part_shift)
        ti.root.dense(ti.i, config.node_num[None]).place(self.node_part_shift_count)
        ti.root.dense(ti.i, config.max_part_num[None]).place(self.part_pid_in_node)
        ti.root.dense(ti.i, config.max_part_num[None]).place(self.part_uid_in_node)

    @ti.kernel
    def clear_node(self, config: ti.template()):
        for i in range(config.node_num[None]):
            self.node_part_count[i] = 0

    @ti.kernel
    def encode(self, obj: ti.template(), config: ti.template()):
        for i in range(obj.part_num[None]):
            obj.neighb_cell_structured_seq[i] = node_encode(obj.pos[i], config)
            obj.neighb_cell_seq[i] = dim_encode(obj.neighb_cell_structured_seq[i], config)
            if 0 < obj.neighb_cell_seq[i] < config.node_num[None]:
                ti.atomic_add(self.node_part_count[obj.neighb_cell_seq[i]], 1)

    @ti.kernel
    def mem_shift(self, config: ti.template()):
        sum = ti.Vector([0])
        for i in range(config.node_num[None]):
            self.node_part_shift[i] = ti.atomic_add(
                sum[0], self.node_part_count[i])
            self.node_part_shift_count[i] = self.node_part_shift[i]

    @ti.kernel
    def fill_node(self, obj: ti.template(), config: ti.template()):
        for i in range(obj.part_num[None]):
            if 0 < obj.neighb_cell_seq[i] < config.node_num[None]:
                obj.neighb_in_cell_seq[i] = atomic_add(
                    self.node_part_shift_count[obj.neighb_cell_seq[i]], 1)
                self.part_pid_in_node[obj.neighb_in_cell_seq[i]] = i
                self.part_uid_in_node[obj.neighb_in_cell_seq[i]] = obj.uid

class Gui():
    def __init__(self, config):
        self.window = ti.ui.Window("Fluid Simulation", (config.gui_res[None][0], config.gui_res[None][1]), vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.make_camera()
        self.camera.position(config.gui_camera_pos[None][0], config.gui_camera_pos[None][1], config.gui_camera_pos[None][2])
        self.camera.lookat(config.gui_camera_lookat[None][0], config.gui_camera_lookat[None][1], config.gui_camera_lookat[None][2])
        self.camera.fov(55)
        self.background_color = (
        (config.gui_canvas_bgcolor[None][0], config.gui_canvas_bgcolor[None][1], config.gui_canvas_bgcolor[None][2]))
        self.ambient_color = (0.7, 0.7, 0.7)
        self.dispaly_radius = config.part_size[1] * 0.5

        # Toggles
        self.show_bound = False
        self.show_help = True
        self.show_run_info = True
        self.op_system_run = False
        self.op_write_file = False
        self.op_refresh_window = True
        self.show_stat = True
        self.show_rod = True
    
    def monitor_listen(self):
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)

        if self.show_help:
            self.window.GUI.begin("options", 0.05, 0.3, 0.2, 0.2)
            self.window.GUI.text("h: help")
            self.window.GUI.text("w: front")
            self.window.GUI.text("s: back")
            self.window.GUI.text("a: left")
            self.window.GUI.text("d: right")
            self.window.GUI.text("RMB: rotate")
            self.window.GUI.text("b: display boundary")
            self.window.GUI.text("r: run system")
            self.window.GUI.text("f: write file")
            self.window.GUI.text("c: refresh window")
            self.window.GUI.end()

        if self.window.get_event(ti.ui.PRESS):
            # run
            if self.window.event.key == 'r':
                self.op_system_run = not self.op_system_run
                print("start to run:", self.op_system_run)

            if self.window.event.key == 'f':
                self.op_write_file = not self.op_write_file
                print("write file:", self.op_write_file)

            if self.window.event.key == 'b':
                self.show_bound = not self.show_bound
                print("show boundary:", self.show_bound)

            if self.window.event.key == 'i':
                self.show_run_info = not self.show_run_info
                print("show run information:", self.show_run_info)

            if self.window.event.key == 'h':
                self.show_help = not self.show_help
                print("show help:", self.show_help)
            
            if self.window.event.key == 'c':
                self.op_refresh_window = not self.op_refresh_window
                print("refresh window:", self.op_refresh_window)

            if self.window.event.key == 'n':
                self.show_rod = not self.show_rod
                print("show rod:", self.show_rod)

            if self.window.event.key == 'x':
                self.show_stat = not self.show_stat
                print("show stat:", self.show_stat)

    def env_set_up(self):
        self.canvas.set_background_color(self.background_color)

    def scene_setup(self):
        self.scene.set_camera(self.camera)
        self.scene.ambient_light(self.ambient_color)
        self.scene.point_light(pos=(2, 1.5, -1.5), color=(0.8, 0.8, 0.8))

    def scene_add_objs(self, obj, radius):
        self.scene.particles(obj.pos_disp, per_vertex_color=obj.color_vector, radius=radius)

    def scene_render(self):
        self.canvas.scene(self.scene)  # Render the scene
        self.window.show()
