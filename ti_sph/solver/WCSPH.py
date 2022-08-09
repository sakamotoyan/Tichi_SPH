# doesn't really use the X-psi notion

import taichi as ti

from ti_sph.solver.SPH_kernel import (
    SPH_kernel,
    bigger_than_zero,
    make_bigger_than_zero,
    spline_W,
    spline_W_old,
    grad_spline_W,
    artificial_Laplacian_spline_W,
)

from ti_sph.solver.SDF_boundary import (
    sdf_contrib,
    sdf_grad_contrib,
)


@ti.data_oriented
class WCSPH(SPH_kernel):
    def __init__(
        self,
        obj,
        dt,
        background_neighb_grid,
        search_template,
        cs,
        gamma=7,
        port_pos="basic.pos",
        port_mass="basic.mass",
        port_rest_volume="basic.rest_volume",
        port_vel="basic.vel",
        port_X="basic.mass",
        port_rest_psi="basic.rest_density",
        port_one="wcsph.one",
        port_sph_psi="wcsph.sph_density",
        port_pressure = "wcsph.pressure",
        port_vel_adv="wcsph.vel_adv",
        port_acc="wcsph.acc",
        port_sph_h="sph.h",
        port_sph_sig="sph.sig",
        port_sph_sig_inv_h="sph.sig_inv_h",
    ):
        self.obj = obj
        self.dt = dt
        self.inv_dt = 1 / dt
        self.background_neighb_grid = background_neighb_grid
        self.search_template = search_template
        self.cs = cs # speed of sound
        self.gamma = gamma

        self.obj_pos = eval("self.obj." + port_pos)
        self.obj_mass = eval("self.obj." + port_mass)
        self.obj_rest_volume = eval("self.obj." + port_rest_volume)
        self.obj_vel = eval("self.obj." + port_vel)
        self.obj_X = eval("self.obj." + port_X)
        self.obj_rest_psi = eval("self.obj." + port_rest_psi)
        self.obj_one = eval("self.obj." + port_one)
        self.obj_sph_psi = eval("self.obj." + port_sph_psi)
        self.obj_pressure = eval("self.obj." + port_pressure)
        self.obj_vel_adv = eval("self.obj." + port_vel_adv)
        self.obj_acc = eval("self.obj." + port_acc)
        self.obj_sph_h = eval("self.obj." + port_sph_h)
        self.obj_sph_sig = eval("self.obj." + port_sph_sig)
        self.obj_sph_sig_inv_h = eval("self.obj." + port_sph_sig_inv_h)

        self.comp_avg_ratio = ti.field(ti.f32, ())  # for statistics

        self.comp_avg_ratio[None] = 1

        self.obj_one.fill(1) # initialize one

        self.kernel_init()

    def update_dt(self, dt):
        self.dt = dt
        self.inv_dt = 1 / dt

    @ti.kernel
    def clear_psi(self):
        for pid in range(self.obj.info.stack_top[None]):
            self.obj_sph_psi[pid] = 0

    @ti.kernel
    def compute_psi_from(
        self,
        from_solver: ti.template(),
    ):
        for pid in range(self.obj.info.stack_top[None]):
            located_cell = from_solver.background_neighb_grid.get_located_cell(
                pos=self.obj_pos[pid],
            )
            for neighb_cell_iter in range(
                from_solver.search_template.get_neighb_cell_num()
            ):
                neighb_cell_index = (
                    from_solver.background_neighb_grid.get_neighb_cell_index(
                        located_cell=located_cell,
                        cell_iter=neighb_cell_iter,
                        neighb_search_template=from_solver.search_template,
                    )
                )
                if from_solver.background_neighb_grid.within_grid(neighb_cell_index):
                    for neighb_part in range(
                        from_solver.background_neighb_grid.get_cell_part_num(
                            neighb_cell_index
                        )
                    ):
                        nid = from_solver.background_neighb_grid.get_neighb_part_id(
                            cell_index=neighb_cell_index,
                            neighb_part_index=neighb_part,
                        )
                        """compute below"""
                        dis = (self.obj_pos[pid] - from_solver.obj_pos[nid]).norm()
                        self.obj_sph_psi[pid] += from_solver.obj_X[nid] * spline_W(
                            dis, self.obj_sph_h[pid], self.obj_sph_sig[pid]
                        )

    @ti.kernel
    def clear_acc(self):
        for pid in range(self.obj.info.stack_top[None]):
            self.obj_acc[pid] *= 0

    @ti.kernel
    def set_vel_adv(self):
        for pid in range(self.obj.info.stack_top[None]):
            self.obj_vel_adv[pid] = self.obj_vel[pid]

    @ti.kernel
    def add_acc(self, acc: ti.template()):
        for pid in range(self.obj.info.stack_top[None]):
            self.obj_acc[pid] += acc[None]

    @ti.kernel
    def update_vel_adv_from_acc(self):
        for pid in range(self.obj.info.stack_top[None]):
            self.obj_vel_adv[pid] += self.obj_acc[pid] * self.dt

    #todo: add directional vis
    def add_acc_from_vis(
        self,
        kinetic_vis_coeff: ti.template(),
        from_solver: ti.template(),
    ):
        self.compute_Laplacian(
            obj=self.obj,
            obj_pos=self.obj_pos,
            nobj_pos=from_solver.obj_pos,
            nobj_volume=from_solver.obj_rest_volume,
            obj_input_attr=self.obj_vel,
            nobj_input_attr=from_solver.obj_vel,
            coeff=kinetic_vis_coeff,
            obj_output_attr=self.obj_acc,
            background_neighb_grid=from_solver.background_neighb_grid,
            search_template=from_solver.search_template,
        )

    @ti.kernel
    def compute_pressure(self):
        for pid in range(self.obj.info.stack_top[None]):
            self.obj_pressure[pid] = self.obj_rest_psi[pid] * self.cs ** 2 / self.gamma \
                * max((self.obj_sph_psi[pid] / self.obj_rest_psi[pid]) ** self.gamma - 1, 0)

    @ti.kernel
    def update_vel_adv_from_pressure(
        self,
        from_solver: ti.template(),
    ):
        for pid in range(self.obj.info.stack_top[None]):
            located_cell = from_solver.background_neighb_grid.get_located_cell(
                pos=self.obj_pos[pid],
            )
            for neighb_cell_iter in range(
                from_solver.search_template.get_neighb_cell_num()
            ):
                neighb_cell_index = (
                    from_solver.background_neighb_grid.get_neighb_cell_index(
                        located_cell=located_cell,
                        cell_iter=neighb_cell_iter,
                        neighb_search_template=from_solver.search_template,
                    )
                )
                if from_solver.background_neighb_grid.within_grid(neighb_cell_index):
                    for neighb_part in range(
                        from_solver.background_neighb_grid.get_cell_part_num(
                            neighb_cell_index
                        )
                    ):
                        nid = from_solver.background_neighb_grid.get_neighb_part_id(
                            cell_index=neighb_cell_index,
                            neighb_part_index=neighb_part,
                        )
                        """compute below"""
                        x_ij = self.obj_pos[pid] - from_solver.obj_pos[nid]
                        dis = x_ij.norm()
                        if bigger_than_zero(dis):
                            self.obj_vel_adv[pid] += -self.dt * self.obj_mass[pid] \
                                * (self.obj_pressure[pid] / self.obj_sph_psi[pid] ** 2 \
                                + from_solver.obj_pressure[nid] / from_solver.obj_sph_psi[nid] ** 2) \
                                * grad_spline_W(dis,
                                        self.obj_sph_h[pid],
                                        self.obj_sph_sig_inv_h[pid],
                                ) * x_ij / dis

    @ti.kernel
    def compute_psi_from_sdf(
        self,
        from_sdf: ti.template(),
    ):
        for pid in range(self.obj.info.stack_top[None]):
            self.obj_sph_psi[pid] += self.obj_rest_psi[pid] * sdf_contrib(self.obj_pos[pid], self.obj_sph_h[pid], from_sdf)

    # update vel_adv from sdf boundary (pressure, friciton, and clamp velocity)
    @ti.kernel
    def update_vel_adv_from_sdf(
        self, 
        from_sdf: ti.template()
    ):
        for pid in range(self.obj.info.stack_top[None]):
            # pressure
            acc_p_bound = -self.obj_rest_psi[pid] \
                * (self.obj_pressure[pid] / self.obj_sph_psi[pid] ** 2) \
                * sdf_grad_contrib(self.obj_pos[pid], self.obj_sph_h[pid], from_sdf)
            # friction
            n = from_sdf.sdf_grad(self.obj_pos[pid])
            vel_b = from_sdf.velocity(self.obj_pos[pid], self.dt)
            vel_rel = self.obj_vel_adv[pid] - vel_b
            v_tan = vel_rel - vel_rel.dot(n) * n #tangential velocity
            a_fric = ti.Vector([0.0,0.0,0.0])
            if v_tan.norm() > 1e-6:
                a_fric = -from_sdf.friction * acc_p_bound.norm() * v_tan.normalized()
            if a_fric.norm() * self.dt > v_tan.norm():
                a_fric = - v_tan / self.dt
            vel_new = self.obj_vel_adv[pid] + self.dt * (acc_p_bound + a_fric)
            # clamp velocity
            vel_rel_new = vel_new - vel_b
            if vel_rel.norm() < vel_rel_new.norm():
                vel_rel_new *= vel_rel.norm() / (vel_rel_new.norm() + 1e-6)
            self.obj_vel_adv[pid] = vel_rel_new + vel_b

    #very simple boundary handling for a box centered at (0,0,0)
    @ti.kernel
    def simpleBoundary(self):
        l = 3.0
        for pid in range(self.obj.info.stack_top[None]):
            for j in ti.static(range(3)):
                if self.obj_pos[pid][j] < -l/2:
                    self.obj_vel_adv[pid][j] = max(0,self.obj_vel_adv[pid][j])
                elif self.obj_pos[pid][j] > l/2:
                    self.obj_vel_adv[pid][j] = min(0,self.obj_vel_adv[pid][j])