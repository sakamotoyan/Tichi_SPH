import taichi as ti
import taichi.math as tm

@ti.kernel
def find_max_vec(self: ti.template(), data: ti.template(), loop_range: ti.i32)->ti.f32:
    tmp_val = 0.0
    for i in range(loop_range):
        ti.atomic_max(tmp_val, tm.length(data[i]))
    return tmp_val

def cfl_dt(self, cfl_factor: float, max_dt: float):
    max_vel = 1e-6
    for part_obj in self.part_obj_list:
        max_vel = max(self.find_max_vec(part_obj.vel, part_obj.get_stack_top()[None]), max_vel)
    new_dt = min(max_dt, self.g_part_size[None] / max_vel * cfl_factor)
    self.set_dt(new_dt)