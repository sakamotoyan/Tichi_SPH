import taichi as ti

def update_stack_top(self, num: int):
    self.m_stack_top[None] += num

def open_stack(self, open_num: ti.template()):
    if self.m_if_stack_open:
        raise Exception("Particle Stack already opened!")
        exit(0)

    if self.m_stack_top[None] + open_num[None] > self.m_part_num[None]:
        raise Exception("Particle Stack overflow!", self.m_stack_top[None], open_num, self.m_part_num[None])
        exit(0)

    self.m_if_stack_open = True
    self.m_stack_open_num[None] = open_num[None]

def fill_open_stack_with_nparr(self, attr_: ti.template(), data: ti.types.ndarray()):
    data_dim = len(data.shape)
    if data_dim == 1:
        data_ti_container = ti.field(ti.f32, self.m_stack_open_num[None])
    elif data_dim == 2:
        data_ti_container = ti.Vector.field(data_dim, ti.f32, self.m_stack_open_num[None])
    else:
        raise Exception("Data dimension not supported!")
        exit(0)
    
    data_ti_container.from_numpy(data)
    self.fill_open_stack_with_arr(attr_, data_ti_container)

@ti.kernel
def fill_open_stack_with_arr(self: ti.template(), attr_: ti.template(), data: ti.template()):
    for i in range(self.m_stack_open_num[None]):
        attr_[i+self.m_stack_top[None]] = data[i]

@ti.kernel
def fill_open_stack_with_val(self: ti.template(), attr_: ti.template(), val: ti.template()):
    for i in range(self.m_stack_open_num[None]):
        attr_[i+self.m_stack_top[None]] = val[None]

def close_stack(self):
    if not self.m_if_stack_open:
        raise Exception("Particle Stack not opened!")
        exit(0)

    self.m_if_stack_open = False
    self.m_stack_top[None] += self.m_stack_open_num[None]
    self.m_stack_open_num[None] = 0


@ti.func
def has_negative(self: ti.template(), val: ti.template()):
    for dim in ti.static(range(self.m_world.dim[None])):
        if val[dim] < 0:
            return True
    return False

@ti.func
def has_positive(self: ti.template(), val: ti.template()):
    for dim in ti.static(range(self.m_world.dim[None])):
        if val[dim] > 0:
            return True
    return False


@ti.kernel
def clear(self: ti.template(), attr_: ti.template()):
    for i in range (self.m_stack_top[None]):
        attr_[i] *= 0

def set_from_numpy(self, to: ti.template(), data: ti.types.ndarray()):
    num = data.shape[0]
    arr = to.to_numpy()
    arr[self.m_stack_top[None]:num, :] = data
    to.from_numpy(arr)

@ti.kernel
def set_val(self: ti.template(), to_arr: ti.template(), num: ti.i32, val: ti.template()):
    for i in range(num):
        to_arr[i+self.m_stack_top[None]] = val[None]