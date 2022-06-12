import taichi as ti
from .DFSPH import *


class DFSPH_layer:
    def __init__(self, solvers, types):
        self.available_types = ["static", "elastic", "fluid"]
        self.dynamic_types = ["elastic", "fluid"]
        self.compressible_types = ["elastic"]

        self.solver_list = solvers
        self.type_list = types
        self.check_pair()

        for solver, type in zip(self.solver_list, self.type_list):
            solver.background_neighb_grid.register(obj_pos=solver.obj_pos)
            if self.is_compressible(type):
                solver.compute_psi_from(solver)
                solver.obj.attr_set_arr(
                    obj_attr=solver.obj_tmp1,
                    val_arr=solver.obj_sph_psi,
                )

    def check_pair(self):
        if len(self.solver_list) != len(self.type_list):
            raise Exception(
                "DF_coupling(): length of solver_list and type_list are not equal."
            )
        for type in self.type_list:
            if not type in self.available_types:
                raise Exception("DF_coupling() does not support '" + type + "' object.")

    def is_dynamic(self, solver_type):
        if solver_type in self.dynamic_types:
            return True
        else:
            return False

    def is_compressible(self, solver_type):
        if solver_type in self.compressible_types:
            return True
        else:
            return False

    def loop(self):
        for solver, type in zip(self.solver_list, self.type_list):

            solver.set_vel_adv()

            solver.clear_psi()
            solver.clear_alpha()
            solver.comp_iter_count[None] = 0

            if self.is_compressible(type):
                solver.compute_psi_from(solver)
                # solver.obj.attr_set_arr(
                #     obj_attr=solver.obj_sph_psi,
                #     val_arr=solver.obj_tmp1,
                # )
                # solver.compute_self_psi()
            else:
                solver.compute_psi_from(solver)

            # loop excluding itself
            for neighbour_solver in self.solver_list:
                if not neighbour_solver is solver:
                    solver.compute_psi_from(neighbour_solver)

            if self.is_dynamic(type):
                # loop including itself
                for neighbour_solver in self.solver_list:
                    solver.compute_alpha_1_from(neighbour_solver)

            # loop including itself
            for neighbour_solver, neighbour_type in zip(
                self.solver_list, self.type_list
            ):
                if self.is_dynamic(neighbour_type):
                    solver.compute_alpha_2_from(neighbour_solver)

            solver.compute_alpha_self()

        while not self.is_global_incompressible():
            self.comp_iter()

        for solver in self.solver_list:
            solver.obj.attr_set_arr(
                obj_attr=solver.obj_vel,
                val_arr=solver.obj_vel_adv,
            )

    def is_global_incompressible(self):
        is_incompressible = True
        for solver in self.solver_list:
            if solver.is_compressible():
                is_incompressible = False
        return is_incompressible

    def comp_iter(self):
        for solver, type in zip(self.solver_list, self.type_list):
            solver.comp_iter_count[None] += 1
            solver.compute_delta_psi_self()

            # if not self.is_compressible(type):
            solver.compute_delta_psi_advection_from(solver)

            # loop excluding itself
            for neighbour_solver in self.solver_list:
                if not neighbour_solver is solver:
                    solver.compute_delta_psi_advection_from(neighbour_solver)

            solver.ReLU_delta_psi()
            solver.check_if_compressible()

        for solver, type in zip(self.solver_list, self.type_list):
            if self.is_dynamic(type):
                # loop including itself
                for neighbour_solver in self.solver_list:
                    solver.update_vel_adv_from(neighbour_solver)
