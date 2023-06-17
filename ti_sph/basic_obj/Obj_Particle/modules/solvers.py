import taichi as ti
from ....basic_solvers.Solver_adv import Adv_slover
from ....basic_solvers.Solver_df import DF_solver

def add_solver_adv(self):
    self.m_solver_adv = Adv_slover(self)

def add_solver_df(self, incompressible_threshold: ti.f32 = 1e-4, div_free_threshold: ti.f32 = 1e-3, incompressible_iter_max: ti.i32 = 100, div_free_iter_max: ti.i32 = 50):
    self.m_solver_df = DF_solver(self, incompressible_threshold, div_free_threshold, incompressible_iter_max, div_free_iter_max)