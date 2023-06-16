import taichi as ti
from ....basic_solvers.Adv_solver import Adv_slover
from ....basic_solvers.DF_solverLayer import DF_solver

def add_solver_adv(self):
    self.m_solver_adv = Adv_slover(self)

def add_solver_df(self):
    self.m_solver_df = DF_solver(self)