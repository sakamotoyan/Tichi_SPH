import taichi as ti

def clear_acc(self):
    for part_obj in self.part_obj_list:
        if (not part_obj.m_is_dynamic) or (part_obj.m_solver_adv) is None:
            continue
        part_obj.m_solver_adv.clear_acc()

def add_acc_gravity(self):
    for part_obj in self.part_obj_list:
        if (not part_obj.m_is_dynamic) or (part_obj.m_solver_adv) is None:
            continue
        part_obj.m_solver_adv.add_acc_gravity()

def acc2vel_adv(self):
    for part_obj in self.part_obj_list:
        if (not part_obj.m_is_dynamic) or (part_obj.m_solver_adv) is None:
            continue
        part_obj.m_solver_adv.acc2vel_adv()

def vel_adv2vel(self):
    for part_obj in self.part_obj_list:
        if (not part_obj.m_is_dynamic) or (part_obj.m_solver_adv) is None:
            continue
        part_obj.m_solver_adv.vel_adv2vel()

def update_pos_from_vel(self):
    for part_obj in self.part_obj_list:
        if (not part_obj.m_is_dynamic) or (part_obj.m_solver_adv) is None:
            continue
        part_obj.m_solver_adv.update_pos()
