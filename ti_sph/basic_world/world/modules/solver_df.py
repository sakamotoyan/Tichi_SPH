import taichi as ti
from typing import List
from ....basic_obj.Obj_Particle import Particle

from ....basic_solvers.Solver_df import DF_solver

def init_solver_df(self):
    self.df_solver_list:List[DF_solver] = []
    for part_obj in self.part_obj_list:
        if part_obj.m_solver_df is not None:
            self.df_solver_list.append(part_obj)
    self.df_incompressible_states: List[bool] = [False for _ in range(len(self.df_solver_list))]
    self.df_divergence_free_states: List[bool] = [False for _ in range(len(self.df_solver_list))]
    
def step_df_incomp(self):
    for part_obj in self.df_solver_list:
        if part_obj.m_is_dynamic:
            part_obj.m_solver_df.get_vel_adv(part_obj.vel_adv)
            self.df_incompressible_states[self.df_solver_list.index(part_obj)] = False
            self.df_divergence_free_states[self.df_solver_list.index(part_obj)] = False
        else:
            self.df_incompressible_states[self.df_solver_list.index(part_obj)] = True
            self.df_divergence_free_states[self.df_solver_list.index(part_obj)] = True

        part_obj.m_solver_df.df_step_static_phase(part_obj.m_neighb_search.neighb_pool)
        
    while True:
        for part_obj in self.df_solver_list:
            part_obj.m_solver_df.incompressible_iter[None] += 1

            part_obj.m_solver_df.compute_delta_density()

            for neighb_obj in part_obj.m_neighb_search.neighb_pool.neighb_obj_list:
                part_obj.m_solver_df.loop_neighb(part_obj.m_neighb_search.neighb_pool, neighb_obj, part_obj.m_solver_df.inloop_update_delta_density_from_vel_adv)
            part_obj.m_solver_df.ReLU_delta_density()
            part_obj.m_solver_df.update_compressible_ratio()

            if part_obj.m_solver_df.compressible_ratio[None] < part_obj.m_solver_df.incompressible_threshold[None] \
                or part_obj.m_solver_df.incompressible_iter[None] > part_obj.m_solver_df.incompressible_iter_max[None]:
                self.df_incompressible_states[self.df_solver_list.index(part_obj)] = True

        for part_obj in self.df_solver_list:
            if part_obj.m_is_dynamic:
                for neighb_obj in part_obj.m_neighb_search.neighb_pool.neighb_obj_list:
                    part_obj.m_solver_df.loop_neighb(part_obj.m_neighb_search.neighb_pool, neighb_obj, part_obj.m_solver_df.inloop_update_vel_adv_from_alpha)
            
        if all(self.df_incompressible_states):
            break

    for part_obj in self.df_solver_list:
        if part_obj.m_is_dynamic:
            part_obj.m_solver_df.update_vel(part_obj.vel)

def step_df_div(self):
    for part_obj in self.df_solver_list:
        if part_obj.m_is_dynamic:
            part_obj.m_solver_df.get_vel_adv(part_obj.vel)
            self.df_divergence_free_states[self.df_solver_list.index(part_obj)] = False
        else:
            self.df_divergence_free_states[self.df_solver_list.index(part_obj)] = True

        part_obj.m_solver_df.df_step_static_phase(part_obj.m_neighb_search.neighb_pool)
        
    while True:
        for part_obj in self.df_solver_list:
            part_obj.m_solver_df.div_free_iter[None] += 1

            part_obj.m_solver_df.compute_delta_density()

            for neighb_obj in part_obj.m_neighb_search.neighb_pool.neighb_obj_list:
                part_obj.m_solver_df.loop_neighb(part_obj.m_neighb_search.neighb_pool, neighb_obj, part_obj.m_solver_df.inloop_update_delta_density_from_vel_adv)
            part_obj.m_solver_df.ReLU_delta_density()
            part_obj.m_solver_df.update_compressible_ratio()
            # print('compressible ratio during', part_obj.m_solver_df.compressible_ratio[None])

            if part_obj.m_solver_df.compressible_ratio[None] < part_obj.m_solver_df.div_free_threshold[None] \
                or part_obj.m_solver_df.div_free_iter[None] > part_obj.m_solver_df.div_free_iter_max[None]:
                self.df_divergence_free_states[self.df_solver_list.index(part_obj)] = True
    
        for part_obj in self.df_solver_list:
            if part_obj.m_is_dynamic:
                for neighb_obj in part_obj.m_neighb_search.neighb_pool.neighb_obj_list:
                    part_obj.m_solver_df.loop_neighb(part_obj.m_neighb_search.neighb_pool, neighb_obj, part_obj.m_solver_df.inloop_update_vel_adv_from_alpha)
        
        if all(self.df_divergence_free_states):
            break

    for part_obj in self.df_solver_list:
        if part_obj.m_is_dynamic:
            part_obj.m_solver_df.update_vel(part_obj.vel)