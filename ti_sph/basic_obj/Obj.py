import taichi as ti
from ..basic_world.World import World
from ..basic_op.type import *

@ti.data_oriented
class Obj:
    def __init__(self, world: World, is_dynamic: bool = True) -> None:
        self.is_dynamic = is_dynamic
        self.attach_to_world(world)
    
    def attach_to_world(self, world: World):
        self.world = world
        # if world dose not have obj_list, create it
        if not hasattr(self.world, "obj_list"):
            self.world.obj_list = []
        if self in self.world.obj_list:
            raise Exception("obj already been attached")
        self.world.obj_list.append(self)
        self.id = val_i(self.world.obj_list.index(self))

    def get_id(self):
        return self.id
    
    @ti.func
    def ti_get_id(self):
        return self.id