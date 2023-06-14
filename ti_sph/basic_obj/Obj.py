import taichi as ti
from ..basic_op.type import *

@ti.data_oriented
class Obj:
    def __init__(self, is_dynamic: bool = True) -> None:
        self.m_is_dynamic = is_dynamic

    def set_id(self, id):
        self.m_id = val_i(id)

    def set_world(self, world):
        self.m_world = world

    def get_id(self):
        return self.m_id

    @ti.func
    def ti_get_id(self):
        return self.m_id