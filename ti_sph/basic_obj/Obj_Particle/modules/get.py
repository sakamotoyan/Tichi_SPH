import taichi as ti

@ti.func
def ti_get_stack_top(self:ti.template()):
    return self.m_stack_top

@ti.func
def ti_get_part_num(self:ti.template()):
    return self.m_part_num

def get_stack_top(self):
    return self.m_stack_top

def get_part_num(self):
    return self.m_part_num

def get_part_size(self):
    return self.m_part_size