import taichi as ti
import numpy as np

@ti.data_oriented
class boxContainer:
    def __init__(self,
        min_corner,
        max_corner,
        friction = 0.1
    ):
        self.min_corner = min_corner
        self.max_corner = max_corner
        self.grad = ti.Vector.field(3, float, (2, 3))
        grad_arr = np.zeros((2, 3, 3), float)
        for i in range(2):
            for j in range(3):
                grad_arr[i, j, j] = 1 - 2 * i
        self.grad.from_numpy(grad_arr)
        self.surface_type = 0
        self.friction = friction

    @ti.func
    def sdf(self, pos_i):
        tmp = 999.0 # arbitrary large number
        d_to_min = pos_i - self.min_corner
        d_to_max = self.max_corner - pos_i
        for j in ti.static(range(3)):
            tmp = min(tmp, min(d_to_min[j], d_to_max[j]))
        return tmp

    @ti.func
    def sdf_grad(self, pos_i):
        tmp = 999.0 # arbitrary large number
        flg = 0
        diam = 0
        d_to_min = pos_i - self.min_corner
        d_to_max = self.max_corner - pos_i
        for j in ti.static(range(3)):
            if d_to_min[j] < tmp:
                tmp = d_to_min[j]
                diam = j
                flg = 0
            if d_to_max[j] < tmp:
                tmp = d_to_max[j]
                diam = j
                flg = 1
        return self.grad[flg, diam]

    @ti.func
    def velocity(self, pos_i, time_step):
        return ti.Vector([0.0, 0.0, 0.0])