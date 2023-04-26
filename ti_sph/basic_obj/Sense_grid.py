import taichi as ti
from typing import List

from .Particle import Particle
from .Obj import Obj

from ..basic_op.type import *
from ..basic_world import World
from ..basic_data_generator.Cube_generator import Cube_generator
from ..basic_neighb_search.Neighb_search import Neighb_search
from ..basic_solvers.DF_solverLayer import DF_solver

DEFAULT = -1

@ti.data_oriented
class Sense_grid(Particle):

    FIXED_GRID = 0
    FIXED_RES = 1
    def __init__(self, 
                 world: World,  # int
                 cell_size: ti.template(), # float | naming logic: the grid is composed of cells, and each cell has a size
                 neighb_pool_size: ti.template(), # int
                 # specifing the type of sense region
                 type: int = FIXED_GRID, # can also be FIXED_RES
                 # Parameters for FIXED_GRID type 
                 grid_lb: ti.template() = DEFAULT, # vector<dim, float> FIXED_GRID
                 grid_rt: ti.template() = DEFAULT, # vector<dim, float> FIXED_GRID
                 # Parameters for FIXED_RES type
                 grid_res: ti.template() = DEFAULT, # val_i FIXED_RES
                 grid_center: ti.template() = DEFAULT, # vector<dim, float> FIXED_RES
                 ):
        # Reading parameters
        self.world = world
        self.neighb_pool_size = neighb_pool_size
        self.cell_size = cell_size
        self.type = type
        self.grid_lb = world.space_lb[None] if grid_lb == DEFAULT else grid_lb[None]
        self.grid_rt = world.space_rt[None] if grid_rt == DEFAULT else grid_rt[None]
        self.grid_res = grid_res
        self.grid_center = grid_center

        self.dim = world.dim
        self.sensed_parts_list: List[Particle] = []

        # Parameters for father classParticle
        self.part_size = self.cell_size
        self.smooth_len = val_f(self.part_size[None] * 2)

        # Though the data for the grid is stored in a one-dimensional way,
        # we can still use the vector type to represent the shape of the grid.
        # And hopefully, accessing the one-dimensional data with the vector type index.
        self.grid_shape = vecx_i(self.dim[None])

        if type == Sense_grid.FIXED_GRID:
            # Parameter check
            if grid_center is not DEFAULT or grid_res is not DEFAULT:
                raise Exception("grid_center and grid_res must NOT be given for FIXED_GRID type")
        if type == Sense_grid.FIXED_RES:
            # Parameter check
            if grid_center is DEFAULT or grid_res is DEFAULT:
                raise Exception("grid_center and grid_res must be given for FIXED_RES type")
            temp_grid_size = self.grid_res[None] * self.cell_size[None]
            temp_grid_lb = self.grid_center[None] - temp_grid_size / 2
            temp_grid_rt = self.grid_center[None] + temp_grid_size / 2
            self.grid_lb = temp_grid_lb
            self.grid_rt = temp_grid_rt
            print("DEBUG grid_lb", self.grid_lb)
            print("DEBUG grid_rt", self.grid_rt)

        self.generator = Cube_generator(self, self.grid_lb, self.grid_rt)
        self.node_num = val_i(self.generator.pushed_num_preview(span=self.get_cell_size()[None]))
        
        super().__init__(part_num=self.get_node_num()[None], world=self.world, part_size=self.get_part_size(), is_dynamic=False)

        self.add_array("pos", vecxf(world.dim[None]).field())
        self.add_array("size", ti.field(ti.f32))
        self.add_array("clampped", ti.field(ti.f32))
        self.add_array("clampped_rgb", vec3f.field())
        # index is for logging the sequence of particles
        self.add_array("index", ti.field(ti.i32), bundle=self.dim[None])

        self.add_attr("density_upper_bound", ti.field(ti.f32, ()))
        self.add_attr("density_lower_bound", ti.field(ti.f32, ()))

        sph = ti.types.struct(
        h=ti.f32,
        sig=ti.f32,
        sig_inv_h=ti.f32,
        density=ti.f32,
        )
        self.add_struct("sph", sph)

        self.generator.push_pos()
        self.generator._get_index(self.index)
        self.set_from_val(to_arr=self.size, num=self.get_node_num()[None], val=self.part_size)
        self.update_stack_top(self.get_node_num()[None])

        self.neighb_search = Neighb_search(self, neighb_pool_size)
        self.df_solver = DF_solver(self)
        
    def add_sensed_particles(self, particles: Particle):
        if particles not in self.sensed_parts_list:
            self.sensed_parts_list.append(particles)
        else:
            raise Exception("particle already added")
        self.neighb_search.add_neighb(particles, self.smooth_len)
        self.neighb_search.search_neighbors()

    def get_cell_size(self):
        return self.get_part_size()
    
    def get_node_num(self):
        return self.node_num
    
    @ti.func
    def ti_get_node_num(self):
        return self.node_num
    
    def step(self):
        self.neighb_search.search_neighbors()
        self.clear(self.sph.density)
        for sensed_parts in self.sensed_parts_list:
            self.df_solver.loop_neighb(self.neighb_search.neighb_pool, sensed_parts, self.df_solver.inloop_accumulate_density)
        self.density_lower_bound[None] = 0
        self.density_upper_bound[None] = 1000
        self.clamp_val(self.sph.density, self.density_lower_bound, self.density_upper_bound)
    
    @ti.kernel
    def clamp_val(self, arr: ti.template(), lower: ti.template(), upper: ti.template()):
        for i in range(self.ti_get_node_num()[None]):
            self.clampped[i] = ti.min(ti.max(arr[i] / (upper[None] - lower[None]),0),1)
            self.clampped_rgb[i] = [self.clampped[i],self.clampped[i],self.clampped[i]]
