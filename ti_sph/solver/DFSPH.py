# handels DFSPH computation and error-related global variables
# PAPER:        Divergence-free smoothed particle hydrodynamics
# AUTHOR:       Jan Bender, Dan Koschier
# CONFERENCE:   SCA '15: Proceedings of the 14th ACM SIGGRAPH / Eurographics Symposium on Computer Animation, August 2015
# URL:          https://doi.org/10.1145/2786784.2786796
# uses the psi-X notation, see (Add url here)

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


@ti.data_oriented
class DFSPH(SPH_kernel):
    def __init__(
        self,
        obj,
        dt,
        background_neighb_grid,
        search_template,
        max_div_error=1e-3,
        max_comp_error=1e-4,
        max_div_iter=50,
        max_comp_iter=100,
        min_comp_tier=2,
        min_div_iter=2,
        port_pos="basic.pos",
        port_mass="basic.mass",
        port_rest_volume="basic.rest_volume",
        port_vel="basic.vel",
        port_X="basic.mass",
        port_rest_psi="basic.rest_density",
        port_one="implicit_sph.one",
        port_sph_psi="implicit_sph.sph_density",
        port_delta_psi="implicit_sph.delta_psi",
        port_alpha_1="implicit_sph.alpha_1",
        port_alpha_2="implicit_sph.alpha_2",
        port_alpha="implicit_sph.alpha",
        port_vel_adv="implicit_sph.vel_adv",
        port_acc="implicit_sph.acc",
        port_sph_h="sph.h",
        port_sph_sig="sph.sig",
        port_sph_sig_inv_h="sph.sig_inv_h",
        port_tmp1="implicit_sph.tmp1",
    ):
        self.obj = obj
        self.dt = dt
        self.inv_dt = 1 / dt
        self.background_neighb_grid = background_neighb_grid
        self.search_template = search_template

        self.obj_pos = eval("self.obj." + port_pos)
        self.obj_mass = eval("self.obj." + port_mass)
        self.obj_rest_volume = eval("self.obj." + port_rest_volume)
        self.obj_vel = eval("self.obj." + port_vel)
        self.obj_X = eval("self.obj." + port_X)
        self.obj_rest_psi = eval("self.obj." + port_rest_psi)
        self.obj_one = eval("self.obj." + port_one)
        self.obj_sph_psi = eval("self.obj." + port_sph_psi)
        self.obj_delta_psi = eval("self.obj." + port_delta_psi)
        self.obj_alpha_1 = eval("self.obj." + port_alpha_1)
        self.obj_alpha_2 = eval("self.obj." + port_alpha_2)
        self.obj_alpha = eval("self.obj." + port_alpha)
        self.obj_vel_adv = eval("self.obj." + port_vel_adv)
        self.obj_acc = eval("self.obj." + port_acc)
        self.obj_sph_h = eval("self.obj." + port_sph_h)
        self.obj_sph_sig = eval("self.obj." + port_sph_sig)
        self.obj_sph_sig_inv_h = eval("self.obj." + port_sph_sig_inv_h)
        self.obj_tmp1 = eval("self.obj." + port_tmp1)

        self.max_div_error = ti.field(ti.f32, ())
        self.max_comp_error = ti.field(ti.f32, ())
        self.max_div_iter = ti.field(ti.i32, ())
        self.max_comp_iter = ti.field(ti.i32, ())
        self.min_comp_tier = ti.field(ti.i32, ())   # typo: tier -> iter
        self.min_div_iter = ti.field(ti.i32, ())
        self.comp_avg_ratio = ti.field(ti.f32, ())  # incompressible error term
        self.div_avg_ratio = ti.field(ti.f32, ())   # divergence-free error term
        self.comp_iter_count = ti.field(ti.i32, ())
        self.div_iter_count = ti.field(ti.i32, ())

        self.max_div_error[None] = float(max_div_error)
        self.max_comp_error[None] = float(max_comp_error)
        self.max_div_iter[None] = int(max_div_iter)
        self.max_comp_iter[None] = int(max_comp_iter)
        self.min_comp_tier[None] = int(min_comp_tier)
        self.min_div_iter[None] = int(min_div_iter)
        self.comp_iter_count[None] = 0
        self.div_iter_count[None] = 0
        self.comp_avg_ratio[None] = 1
        self.div_avg_ratio[None] = 1
        self.obj_one.fill(1)

        self.kernel_init()

    # returns whether incompressible solver iteration should be continued
    def is_compressible(self):
        return (
            (self.comp_iter_count[None] < self.min_comp_tier[None])
            or (self.comp_avg_ratio[None] > self.max_comp_error[None])
            and (not self.comp_iter_count[None] == self.max_comp_iter[None])
        )

    def is_div(self):
        return (
            (self.div_iter_count[None] < self.min_div_iter[None])
            or (self.div_avg_ratio[None] > self.max_div_error[None])
            and (not self.div_iter_count[None] == self.max_div_iter[None])
        )

    def update_dt(self, dt):
        self.dt = dt
        self.inv_dt = 1 / dt

    #compute psi (corresponds to SPH density estimationn)
    @ti.kernel
    def compute_psi(
        self,
        obj: ti.template(),
        obj_pos: ti.template(),
        nobj_pos: ti.template(),
        nobj_X: ti.template(),
        obj_output_psi: ti.template(),
        background_neighb_grid: ti.template(),
        search_template: ti.template(),
    ):
        for pid in range(obj.info.stack_top[None]):
            located_cell = background_neighb_grid.get_located_cell(
                pos=obj_pos[pid],
            )
            for neighb_cell_iter in range(search_template.get_neighb_cell_num()):
                neighb_cell_index = background_neighb_grid.get_neighb_cell_index(
                    located_cell=located_cell,
                    cell_iter=neighb_cell_iter,
                    neighb_search_template=search_template,
                )
                if background_neighb_grid.within_grid(neighb_cell_index):
                    for neighb_part in range(
                        background_neighb_grid.get_cell_part_num(neighb_cell_index)
                    ):
                        nid = background_neighb_grid.get_neighb_part_id(
                            cell_index=neighb_cell_index,
                            neighb_part_index=neighb_part,
                        )
                        """compute below"""
                        dis = (obj_pos[pid] - nobj_pos[nid]).norm()
                        obj_output_psi[pid] += nobj_X[nid] * spline_W(
                            dis, obj.sph.h[pid], obj.sph.sig[pid]
                        )

    @ti.kernel
    def compute_self_psi(self):
        for pid in range(self.obj.info.stack_top[None]):
            self.obj_sph_psi[pid] += self.obj_X[pid] * spline_W(
                0, self.obj_sph_h[pid], self.obj_sph_sig[pid]
            )

    # compute alpha_1
    @ti.kernel
    def compute_alpha_1(
        self,
        obj: ti.template(),
        obj_pos: ti.template(),
        nobj_pos: ti.template(),
        nobj_X: ti.template(),
        obj_output_alpha_1: ti.template(),
        background_neighb_grid: ti.template(),
        search_template: ti.template(),
    ):
        for pid in range(obj.info.stack_top[None]):
            located_cell = background_neighb_grid.get_located_cell(
                pos=obj_pos[pid],
            )
            for neighb_cell_iter in range(search_template.get_neighb_cell_num()):
                neighb_cell_index = background_neighb_grid.get_neighb_cell_index(
                    located_cell=located_cell,
                    cell_iter=neighb_cell_iter,
                    neighb_search_template=search_template,
                )
                if background_neighb_grid.within_grid(neighb_cell_index):
                    for neighb_part in range(
                        background_neighb_grid.get_cell_part_num(neighb_cell_index)
                    ):
                        nid = background_neighb_grid.get_neighb_part_id(
                            cell_index=neighb_cell_index,
                            neighb_part_index=neighb_part,
                        )
                        """compute below"""
                        x_ij = obj_pos[pid] - nobj_pos[nid]
                        dis = x_ij.norm()
                        if bigger_than_zero(dis):
                            grad_W_vec = (
                                grad_spline_W(
                                    dis, obj.sph.h[pid], obj.sph.sig_inv_h[pid]
                                )
                                * x_ij
                                / dis
                            )
                            obj_output_alpha_1[pid] += nobj_X[nid] * grad_W_vec

    # compute alpha_2
    @ti.kernel
    def compute_alpha_2(
        self,
        obj: ti.template(),
        obj_pos: ti.template(),
        nobj_pos: ti.template(),
        nobj_mass: ti.template(),
        nobj_X: ti.template(),
        obj_output_alpha_2: ti.template(),
        background_neighb_grid: ti.template(),
        search_template: ti.template(),
    ):
        for pid in range(obj.info.stack_top[None]):
            located_cell = background_neighb_grid.get_located_cell(
                pos=obj_pos[pid],
            )
            for neighb_cell_iter in range(search_template.get_neighb_cell_num()):
                neighb_cell_index = background_neighb_grid.get_neighb_cell_index(
                    located_cell=located_cell,
                    cell_iter=neighb_cell_iter,
                    neighb_search_template=search_template,
                )
                if background_neighb_grid.within_grid(neighb_cell_index):
                    for neighb_part in range(
                        background_neighb_grid.get_cell_part_num(neighb_cell_index)
                    ):
                        nid = background_neighb_grid.get_neighb_part_id(
                            cell_index=neighb_cell_index,
                            neighb_part_index=neighb_part,
                        )
                        """compute below"""
                        x_ij = obj_pos[pid] - nobj_pos[nid]
                        dis = x_ij.norm()
                        if bigger_than_zero(dis):
                            grad_W = grad_spline_W(
                                dis, obj.sph.h[pid], obj.sph.sig_inv_h[pid]
                            )
                            obj_output_alpha_2[pid] += (
                                nobj_X[nid] * grad_W
                            ) ** 2 / nobj_mass[nid]

    # compute alpha
    @ti.kernel
    def compute_alpha(
        self,
        obj: ti.template(),
        obj_mass: ti.template(),
        obj_alpha_1: ti.template(),
        obj_alpha_2: ti.template(),
        obj_output_alpha: ti.template(),
    ):
        for pid in range(obj.info.stack_top[None]):
            obj_output_alpha[pid] = (
                obj_alpha_1[pid].dot(obj_alpha_1[pid]) / obj_mass[pid]
            ) + obj_alpha_2[pid]
            if not bigger_than_zero(obj_output_alpha[pid]):
                obj_output_alpha[pid] = make_bigger_than_zero()

    # compute delta_psi
    @ti.kernel
    def compute_delta_psi(
        self,
        obj: ti.template(),
        obj_sph_psi: ti.template(),
        obj_rest_psi: ti.template(),
        obj_output_delta_psi: ti.template(),
    ):
        for pid in range(obj.info.stack_top[None]):
            obj_output_delta_psi[pid] = obj_sph_psi[pid] - obj_rest_psi[pid]

    # compute delta_psi from advection
    @ti.kernel
    def compute_adv_psi_advection(
        self,
        obj: ti.template(),
        obj_pos: ti.template(),
        obj_vel_adv: ti.template(),
        nobj_pos: ti.template(),
        nobj_vel_adv: ti.template(),
        nobj_X: ti.template(),
        dt: ti.template(),
        obj_output_delta_psi: ti.template(),
        background_neighb_grid: ti.template(),
        search_template: ti.template(),
    ):
        for pid in range(obj.info.stack_top[None]):
            located_cell = background_neighb_grid.get_located_cell(
                pos=obj_pos[pid],
            )
            for neighb_cell_iter in range(search_template.get_neighb_cell_num()):
                neighb_cell_index = background_neighb_grid.get_neighb_cell_index(
                    located_cell=located_cell,
                    cell_iter=neighb_cell_iter,
                    neighb_search_template=search_template,
                )
                if background_neighb_grid.within_grid(neighb_cell_index):
                    for neighb_part in range(
                        background_neighb_grid.get_cell_part_num(neighb_cell_index)
                    ):
                        nid = background_neighb_grid.get_neighb_part_id(
                            cell_index=neighb_cell_index,
                            neighb_part_index=neighb_part,
                        )
                        """compute below"""
                        x_ij = obj_pos[pid] - nobj_pos[nid]
                        dis = x_ij.norm()
                        if bigger_than_zero(dis):
                            obj_output_delta_psi[pid] += (
                                (
                                    grad_spline_W(
                                        dis, obj.sph.h[pid], obj.sph.sig_inv_h[pid]
                                    )
                                    * x_ij
                                    / dis
                                ).dot(obj_vel_adv[pid] - nobj_vel_adv[nid])
                                * nobj_X[nid]
                                * dt[None]
                            )

    # clamp delta_psi to above 0, AND update comp_avg_ratio (error term)
    @ti.kernel
    def statistic_non_negative_delta_psi(
        self,
        obj: ti.template(),
        obj_rest_psi: ti.template(),
        obj_output_delta_psi: ti.template(),
    ):
        for i in range(obj.info.stack_top[None]):
            if obj_output_delta_psi[i] < 0:
                obj_output_delta_psi[i] = 0
        for i in range(obj.info.stack_top[None]):
            self.comp_avg_ratio[None] += obj_output_delta_psi[i] / obj_rest_psi[i]
        self.comp_avg_ratio[None] /= obj.info.stack_top[None]

    # update vel_adv with pressure
    @ti.kernel
    def ReLU(self, attr: ti.template()):
        for pid in range(self.obj.info.stack_top[None]):
            if attr[pid] < 0:
                attr[pid] = 0

    def ReLU_delta_psi(self):
        self.ReLU(self.obj_delta_psi)

    @ti.kernel
    def check_if_compressible(self):
        for pid in range(self.obj.info.stack_top[None]):
            self.comp_avg_ratio[None] += (
                self.obj_delta_psi[pid] / self.obj_rest_psi[pid]
            )

        self.comp_avg_ratio[None] /= self.obj.info.stack_top[None]

    @ti.kernel
    def check_if_divfree(self):
        for pid in range(self.obj.info.stack_top[None]):
            self.div_avg_ratio[None] += (
                self.obj_delta_psi[pid] / self.obj_rest_psi[pid]
            )

        self.div_avg_ratio[None] /= self.obj.info.stack_top[None]

    @ti.kernel
    def update_vel_adv(
        self,
        obj: ti.template(),
        obj_pos: ti.template(),
        obj_X: ti.template(),
        obj_delta_psi: ti.template(),
        obj_alpha: ti.template(),
        obj_mass: ti.template(),
        nobj_pos: ti.template(),
        nobj_delta_psi: ti.template(),
        nobj_X: ti.template(),
        nobj_alpha: ti.template(),
        inv_dt: ti.template(),
        obj_output_vel_adv: ti.template(),
        background_neighb_grid: ti.template(),
        search_template: ti.template(),
    ):
        for pid in range(obj.info.stack_top[None]):
            located_cell = background_neighb_grid.get_located_cell(
                pos=obj_pos[pid],
            )
            for neighb_cell_iter in range(search_template.get_neighb_cell_num()):
                neighb_cell_index = background_neighb_grid.get_neighb_cell_index(
                    located_cell=located_cell,
                    cell_iter=neighb_cell_iter,
                    neighb_search_template=search_template,
                )
                if background_neighb_grid.within_grid(neighb_cell_index):
                    for neighb_part in range(
                        background_neighb_grid.get_cell_part_num(neighb_cell_index)
                    ):
                        nid = background_neighb_grid.get_neighb_part_id(
                            cell_index=neighb_cell_index,
                            neighb_part_index=neighb_part,
                        )
                        """compute below"""
                        x_ij = obj_pos[pid] - nobj_pos[nid]
                        dis = x_ij.norm()
                        if bigger_than_zero(dis):
                            obj_output_vel_adv[pid] += (
                                -inv_dt[None]
                                * (
                                    grad_spline_W(
                                        dis, obj.sph.h[pid], obj.sph.sig_inv_h[pid]
                                    )
                                    * x_ij
                                    / dis
                                )
                                / obj_mass[pid]
                                * (
                                    (obj_delta_psi[pid] * nobj_X[nid] / obj_alpha[pid])
                                    + (
                                        nobj_delta_psi[nid]
                                        * obj_X[pid]
                                        / nobj_alpha[nid]
                                    )
                                )
                            )

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
    def compute_number_density_psi_from(
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
                        self.obj_sph_psi[pid] += self.obj_X[pid] * spline_W(
                            dis, self.obj_sph_h[pid], self.obj_sph_sig[pid]
                        )

    @ti.kernel
    def compute_alpha_1_from(
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
                            grad_W_vec = (
                                grad_spline_W(
                                    dis,
                                    self.obj_sph_h[pid],
                                    self.obj_sph_sig_inv_h[pid],
                                )
                                * x_ij
                                / dis
                            )
                            self.obj_alpha_1[pid] += from_solver.obj_X[nid] * grad_W_vec

    @ti.kernel
    def compute_alpha_2_from(
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
                            grad_W = grad_spline_W(
                                dis,
                                self.obj_sph_h[pid],
                                self.obj_sph_sig_inv_h[pid],
                            )
                            self.obj_alpha_2[pid] += (
                                from_solver.obj_X[nid] * grad_W
                            ) ** 2 / from_solver.obj_mass[nid]

    @ti.kernel
    def compute_alpha_self(
        self,
    ):
        for pid in range(self.obj.info.stack_top[None]):
            self.obj_alpha[pid] = (
                self.obj_alpha_1[pid].dot(self.obj_alpha_1[pid]) / self.obj_mass[pid]
            ) + self.obj_alpha_2[pid]
            if not bigger_than_zero(self.obj_alpha[pid]):
                self.obj_alpha[pid] = make_bigger_than_zero()

    @ti.kernel
    def compute_delta_psi_self(
        self,
    ):
        for pid in range(self.obj.info.stack_top[None]):
            self.obj_delta_psi[pid] = self.obj_sph_psi[pid] - self.obj_rest_psi[pid]

    @ti.kernel
    def compute_delta_psi_advection_from(
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
                            self.obj_delta_psi[pid] += (
                                (
                                    grad_spline_W(
                                        dis,
                                        self.obj_sph_h[pid],
                                        self.obj_sph_sig_inv_h[pid],
                                    )
                                    * x_ij
                                    / dis
                                ).dot(
                                    self.obj_vel_adv[pid] - from_solver.obj_vel_adv[nid]
                                )
                                * from_solver.obj_X[nid]
                                * self.dt
                            )
    @ti.kernel
    def compute_delta_numbder_density_psi_advection_from(
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
                            self.obj_delta_psi[pid] += (
                                (
                                    grad_spline_W(
                                        dis,
                                        self.obj_sph_h[pid],
                                        self.obj_sph_sig_inv_h[pid],
                                    )
                                    * x_ij
                                    / dis
                                ).dot(
                                    self.obj_vel_adv[pid] - from_solver.obj_vel_adv[nid]
                                )
                                * self.obj_X[pid]
                                * self.dt
                            )

    @ti.kernel
    def update_vel_adv_from(
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
                            self.obj_vel_adv[pid] += (
                                -self.inv_dt
                                * (
                                    grad_spline_W(
                                        dis,
                                        self.obj_sph_h[pid],
                                        self.obj_sph_sig_inv_h[pid],
                                    )
                                    * x_ij
                                    / dis
                                )
                                / self.obj_mass[pid]
                                * (
                                    (
                                        self.obj_delta_psi[pid]
                                        * from_solver.obj_X[nid]
                                        / self.obj_alpha[pid]
                                    )
                                    + (
                                        from_solver.obj_delta_psi[nid]
                                        * self.obj_X[pid]
                                        / from_solver.obj_alpha[nid]
                                    )
                                )
                            )

    @ti.kernel
    def clear_alpha(self):
        for pid in range(self.obj.info.stack_top[None]):
            self.obj_alpha_1[pid] *= 0
            self.obj_alpha_2[pid] *= 0
            self.obj_alpha[pid] *= 0

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
            self.obj_acc[pid] = acc[None]

    @ti.kernel
    def update_vel_adv_from_acc(self):
        for pid in range(self.obj.info.stack_top[None]):
            self.obj_vel_adv[pid] += self.obj_acc[pid] * self.dt
    
    @ti.kernel
    def update_vel_from_acc(self):
        for pid in range(self.obj.info.stack_top[None]):
            self.obj_vel[pid] += self.obj_acc[pid] * self.dt

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

    # @ti.kernel
    # def update_force(
    #     self,
    #     obj: ti.template(),
    #     obj_pos: ti.template(),
    #     obj_X: ti.template(),
    #     obj_delta_psi: ti.template(),
    #     obj_alpha: ti.template(),
    #     nobj_pos: ti.template(),
    #     nobj_delta_psi: ti.template(),
    #     nobj_X: ti.template(),
    #     nobj_alpha: ti.template(),
    #     inv_dt: ti.template(),
    #     obj_output_force: ti.template(),
    #     background_neighb_grid: ti.template(),
    #     search_template: ti.template(),
    # ):
    #     for pid in range(obj.info.stack_top[None]):
    #         located_cell = background_neighb_grid.get_located_cell(
    #             pos=obj_pos[pid],
    #         )
    #         for neighb_cell_iter in range(search_template.get_neighb_cell_num()):
    #             neighb_cell_index = background_neighb_grid.get_neighb_cell_index(
    #                 located_cell=located_cell,
    #                 cell_iter=neighb_cell_iter,
    #                 neighb_search_template=search_template,
    #             )
    #             if background_neighb_grid.within_grid(neighb_cell_index):
    #                 for neighb_part in range(
    #                     background_neighb_grid.get_cell_part_num(neighb_cell_index)
    #                 ):
    #                     nid = background_neighb_grid.get_neighb_part_id(
    #                         cell_index=neighb_cell_index,
    #                         neighb_part_index=neighb_part,
    #                     )
    #                     """compute below"""
    #                     x_ij = obj_pos[pid] - nobj_pos[nid]
    #                     dis = x_ij.norm()
    #                     if bigger_than_zero(dis):
    #                         obj_output_force[pid] += (
    #                             -ti.pow(inv_dt[None], 2)
    #                             * (
    #                                 grad_spline_W(
    #                                     dis, obj.sph.h[pid], obj.sph.sig_inv_h[pid]
    #                                 )
    #                                 * x_ij
    #                                 / dis
    #                             )
    #                             * (
    #                                 (obj_delta_psi[pid] * nobj_X[nid] / obj_alpha[pid])
    #                                 + (
    #                                     nobj_delta_psi[nid]
    #                                     * obj_X[pid]
    #                                     / nobj_alpha[nid]
    #                                 )
    #                             )
    #                         )

    # / update acc /
    # @ti.kernel
    # def update_acc(
    #     self,
    #     input_force: ti.template(),
    #     output_acc: ti.template(),
    # ):
    #     for pid in range(self.obj.info.stack_top[None]):
    #         output_acc[pid] += input_force[pid] / self.obj_mass[pid]
