import taichi as ti

from .Particle import Particle
from .Obj import Obj

from ..basic_op.type import *
from ..basic_world import World
from ..basic_data_generator.cube import Cube_generator
from .. basic_neighb_search.neighb_search import Neighb_search
from ..basic_solvers.DFSPH_solver import DF_solver

DEFAULT = -1

@ti.data_oriented
class Cartesian(Particle):

    FIXED_GRID = 0
    FIXED_RES = 1
    def __init__(self, 
                 world: World,  # int
                 sense_region_grid_size: ti.template(), # float
                 sense_region_lb: ti.template() = DEFAULT, # vector<dim, float>
                 sense_region_rt: ti.template() = DEFAULT, # vector<dim, float>
                 sense_region_size: ti.template() = DEFAULT, # vector<dim, float>
                 sense_region_res: ti.template() = DEFAULT, # vector<dim, float>
                 type: int = FIXED_GRID,
                 ):
        self.world = world
        self.dim = world.dim

        if type == Cartesian.FIXED_GRID:

            self.part_size = sense_region_grid_size
            self.smooth_len = val_f(self.part_size[None] * 2)

            if sense_region_lb == DEFAULT:
                self.sense_region_lb = world.space_lb[None]
            else:
                self.sense_region_lb = sense_region_lb[None]
            
            if sense_region_rt == DEFAULT:
                self.sense_region_rt = world.space_rt[None]
            else:
                self.sense_region_rt = sense_region_rt[None]

            self.generator = Cube_generator(self, self.sense_region_lb, self.sense_region_rt)
            self.node_num = val_i(self.generator.pushed_num_preview(res=self.get_grid_size()[None]))
        
        if type == Cartesian.FIXED_RES:
            if sense_region_size == DEFAULT or sense_region_res == DEFAULT:
                raise Exception("sense_region_size and sense_region_res must be given")
            
            raise Exception("not implemented")
            self.part_size = val_f(sense_region_size[None]/sense_region_res[None])

        super().__init__(part_num=self.get_node_num()[None], world=world, part_size=self.get_part_size(), is_dynamic=False)
        self.add_array("pos", vecxf(world.dim[None]).field())
        self.add_array("size", ti.field(ti.f32))
        self.add_array("clampped", ti.field(ti.f32))
        self.add_array("clampped_rgb", vec3f.field())

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
        self.set_from_val(to_arr=self.size, num=self.get_node_num()[None], val=self.part_size)
        self.update_stack_top(self.get_node_num()[None])

        self.neighb_search = Neighb_search(self, val_i(1e6))
        self.df_solver = DF_solver(self)
        
    def add_sensed_particles(self, particles: Particle):
        self.sensed_parts = particles
        self.neighb_search.add_neighb(particles, self.smooth_len)
        self.neighb_search.search_neighbors()

    def get_grid_size(self):
        return self.get_part_size()
    
    def get_node_num(self):
        return self.node_num
    
    def step(self):
        self.neighb_search.search_neighbors()
        self.clear(self.sph.density)
        self.df_solver.loop_neighb(self.neighb_search.neighb_pool, self.sensed_parts, self.df_solver.inloop_accumulate_density)
        self.density_lower_bound[None] = 0
        self.density_upper_bound[None] = 1000
        self.clamp_val(self.sph.density, self.density_lower_bound, self.density_upper_bound)
    
    @ti.kernel
    def clamp_val(self, arr: ti.template(), lower: ti.template(), upper: ti.template()):
        for i in range(self.get_node_num()[None]):
            self.clampped[i] = ti.min(ti.max(arr[i] / (upper[None] - lower[None]),0),1)
            self.clampped_rgb[i] = [self.clampped[i],self.clampped[i],self.clampped[i]]