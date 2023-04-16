import taichi as ti
from ..basic_op.type import *
from .Neighb_pool import Neighb_pool
from .neighb_cell_simple import Neighb_cell_simple
from ..basic_obj.Particle import Particle

class Neighb_search:
    def __init__(self, obj:Particle, max_neighb_num:ti.template()=0):
        self.attach_to_obj(obj)

        self.neighb_cell = Neighb_cell_simple(obj)
        self.neighb_pool = Neighb_pool(obj, max_neighb_num)

    def attach_to_obj(self, obj):
        self.obj = obj
        if not hasattr(obj, "neighb_search"):
            obj.neighb_search = self
        else:
            raise Exception("obj already has neighb_search")
        
    def add_neighb(self, neighb_obj: Particle, search_range: ti.template()):
        self.neighb_pool.add_neighb_obj(neighb_obj, search_range)

    def update_self(self):
        self.neighb_cell.update_part_in_cell()

    def search_neighbors(self):
        self.neighb_pool.register_neighbours()