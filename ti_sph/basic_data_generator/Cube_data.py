import taichi as ti
import numpy as np

from .Data_generator import Data_generator

@ti.data_oriented
class Cube_data(Data_generator):
    def __init__(self, lb: ti.Vector, rt: ti.Vector, span: float):
        self.lb = lb
        self.rt = rt
        self.span = span
        self.dim = len(lb)

        self.shape = np.ceil((self.rt - self.lb) / span).astype(np.int32) # the number of voxels in each dimension
        pos_frac = [] # the position of each voxel in each dimension
        index_frac = [] # corresponding index of elements in pos_frac 

        for i in range(self.dim):
            pos_frac.append(np.linspace(self.lb[i], self.lb[i]+span*self.shape[i], self.shape[i]+1))
            index_frac.append(np.linspace(0,self.shape[i], self.shape[i]+1).astype(np.int32))
        
        # returned values
        self.pos = np.array(np.meshgrid(*pos_frac)).T.reshape(-1, self.dim) # the pos array takes the form of (num, dim)
        self.index = np.array(np.meshgrid(*index_frac)).T.reshape(-1, self.dim) # the index array takes the form of (num, dim)
        self.num = self.pos.shape[0]
    
    def translate(self, offset: ti.Vector):
        self.pos += offset.to_numpy()
        return self


# debug_cube_data = Cube_data(ti.Vector([0,0,0]), ti.Vector([1,1,1]), 0.1)
# # print(debug_cube_data.pos)
# debug_cube_data.translate(ti.Vector([1,1,1]))
# print(debug_cube_data.pos)