import taichi as ti
from ti_sph.basic_op import *
from ti_sph import *
import numpy as np

def grid_template(world, grid_size):
    proj_grid = Cartesian(type=Cartesian.FIXED_GRID, world=world, sense_region_grid_size=val_f(grid_size))

   

    return proj_grid
