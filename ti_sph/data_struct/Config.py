import taichi as ti
from .struct.constructor import *


@ti.data_oriented
class Config:
    def __init__(self, dim, capacity_list):
        info_construct(self, dim, capacity_list)
