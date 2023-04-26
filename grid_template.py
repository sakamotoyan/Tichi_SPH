import taichi as ti
from ti_sph.basic_op import *
from ti_sph import *
import numpy as np

def grid_template(world, grid_size):
    proj_grid = Sense_grid(type=Sense_grid.FIXED_GRID, world=world, cell_size=val_f(grid_size))

   

    return proj_grid
