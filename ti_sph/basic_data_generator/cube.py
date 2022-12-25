import taichi as ti
import numpy as np

class Cube_generator:
    def __init__(self, lb:np.array, rt:np.array, span:float):
        self.lb = lb
        self.rt = rt
        self.span = span
        self.dim = len(lb)
    
    def __init__(self, lb:ti.Vector, rt:ti.Vector, span:float):
        self.lb = lb.to_numpy()
        self.rt = rt.to_numpy()
        self.span = span
        self.dim = len(lb)
    
    def generate_pos_based_on_span(self):
        voxel_num = np.ceil((self.rt - self.lb) / self.span)
        voxel_num = voxel_num.astype(np.int32)
        
        new_voxel_num = np.zeros(self.dim, dtype=np.int32)
        pos_frac = []
        for i in range(self.dim):
            pos_frac.append(np.linspace(self.lb[i], self.lb[i]+self.span*voxel_num[i], voxel_num[i]+1))
            # print(pos_frac[i])
            new_voxel_num[i] = len(pos_frac[i])
        new_voxel_total_num = np.prod(new_voxel_num)

        pos = np.zeros(np.append(new_voxel_num, self.dim), dtype=np.float32)
        

        cartesian_product = np.array(np.meshgrid(*pos_frac)).T.reshape(-1, self.dim)
        return cartesian_product
        # print(cartesian_product.shape)

        # for dim in range(self.dim):
        #     print(np.linspace(self.lb[dim], self.rt[dim], voxel_num[dim]))
            # pos[:,dim] = np.linspace(self.lb[dim], self.rt[dim], voxel_num[dim])

        # for i in range(np.prod(voxel_num)):
        #     print(i)
        #     pos[:,i] = np.linspace(self.lb[i], self.rt[i], voxel_num[i])


        # for i,j,k in np.ndindex(voxel_num):
        #     pos[i,j,k] = np.linspace(self.lb[0], self.rt[0], voxel_num[0])[i]

        # for i in range(self.dim):
        #     pos[:,i] = np.linspace(self.lb[i], self.rt[i], voxel_num[i])
        
        pos = pos.reshape(new_voxel_total_num, self.dim)
        print("pos:", pos.shape)


