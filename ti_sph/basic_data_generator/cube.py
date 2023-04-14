import taichi as ti
import numpy as np

from .generator import *

@ti.data_oriented
class Cube_generator(Data_generator):

    num = 0

    def __init__(self, obj, lb: np.array, rt: np.array):
        self.obj = obj
        self.lb = lb
        self.rt = rt
        self.dim = len(lb)

    def __init__(self, obj, lb: ti.Vector, rt: ti.Vector):
        self.obj = obj
        self.lb = lb.to_numpy()
        self.rt = rt.to_numpy()
        self.dim = len(lb)

    def generate_pos_based_on_span(self, span: float):
        voxel_num = np.ceil((self.rt - self.lb) / span)
        voxel_num = voxel_num.astype(np.int32)

        pos_frac = []
        for i in range(self.dim):
            pos_frac.append(np.linspace(
                self.lb[i], self.lb[i]+span*voxel_num[i], voxel_num[i]+1))

        self.pos_arr = np.array(np.meshgrid(*pos_frac)).T.reshape(-1, self.dim)
        self.num = self.pos_arr.shape[0]
        
        # alias for pos_arr
        self.np_pos = self.pos_arr

        return (self.pos_arr, self.num)

    def push_pos(
            self, 
            span: float = -1,
            ):
        if span < 0:
            span = self.obj.part_size[None]
        self.generate_pos_based_on_span(span)
        cube_pos_data = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.num)
        cube_pos_data.from_numpy(self.np_pos)
        self.ker_push_pos_based_on_span(cube_pos_data, self.obj.pos, self.obj.stack_top)
        return self.num
    
    @ti.kernel
    def ker_push_pos_based_on_span(self, cube_pos_data:ti.template(), pos_arr:ti.template(), stack_top:ti.template()):
        for i in range(self.num):
            pos_arr[stack_top[None]+i] = cube_pos_data[i]
        
