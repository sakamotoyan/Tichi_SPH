import taichi as ti

from .modules import attr_allo
from .modules import data_op
from .modules import get
from .modules import neighb_search
from .modules import solvers

from ..Obj import Obj
from ...basic_op.type import *

@ti.data_oriented
class Particle(Obj):
    def __init__(
        self,
        part_num: int,
        part_size: ti.template(),
        is_dynamic: bool = True,
    ):
        self.m_world = None
        self.m_id = None    
        super().__init__(is_dynamic)

        if (part_num <= 0):
            raise ValueError("part_num must be larger than 0")

        # data structure management
        self.m_part_num = val_i(part_num)
        self.m_stack_top = val_i(0)
        self.m_if_stack_open = False
        self.m_stack_open_num = val_i(0)
        # shared attributes
        self.m_part_size = part_size

        # modules
        self.m_neighb_search = None
        self.m_solver_adv = None
        self.m_solver_df = None

        # data structure
        self.m_attr_list = {}
        self.m_array_list = {}
        self.m_struct_list = {}

        # TODO
        self.m_delete_list = ti.field(ti.i32, self.m_part_num[None]+1)

    # Functions: attribute related
    add_attr = attr_allo.add_attr
    add_attrs = attr_allo.add_attrs
    add_array = attr_allo.add_array
    add_arrays = attr_allo.add_arrays
    add_struct = attr_allo.add_struct
    add_structs = attr_allo.add_structs
    instantiate_from_template = attr_allo.instantiate_from_template

    # Functions: verbose functions
    verbose_structs = attr_allo.verbose_structs
    verbose_arrays = attr_allo.verbose_arrays
    verbose_attrs = attr_allo.verbose_attrs
    
    # Functions: Data operations
    update_stack_top = data_op.update_stack_top
    open_stack = data_op.open_stack
    fill_open_stack_with_nparr = data_op.fill_open_stack_with_nparr
    fill_open_stack_with_arr = data_op.fill_open_stack_with_arr
    fill_open_stack_with_val = data_op.fill_open_stack_with_val
    close_stack = data_op.close_stack
    clear = data_op.clear
    set_from_numpy = data_op.set_from_numpy
    set_val = data_op.set_val
    has_negative = data_op.has_negative
    has_positive = data_op.has_positive

    # Functions: Data access for single values
    get_stack_top = get.get_stack_top
    get_part_num = get.get_part_num
    get_part_size = get.get_part_size
    ti_get_stack_top = get.ti_get_stack_top
    ti_get_part_num = get.ti_get_part_num

    # Functions: neighb search related
    add_module_neighb_search = neighb_search.add_neighb_search
    check_neighb_search = neighb_search.check_neighb_search
    add_neighb_obj = neighb_search.add_neighb_obj
    add_neighb_objs = neighb_search.add_neighb_objs
    add_solver_adv = solvers.add_solver_adv
    add_solver_df = solvers.add_solver_df
    
    # TODO
    def delete_outbounded_particles(self):
        self.clear(self.m_delete_list)
        self.log_tobe_deleted_particles()
    # TODO
    @ti.kernel
    def log_tobe_deleted_particles(self):
        counter = ti.static(self.m_delete_list[self.m_delete_list.shape[0]])
        for part_id in range(self.m_stack_top[None]):
            if self.has_negative(self.pos[part_id]-self.m_world.space_lb[None]) or self.has_positive(self.pos[part_id]-self.m_world.space_rt[None]):
                self.m_delete_list[ti.atomic_add(self.m_delete_list[counter],1)] = part_id
    # TODO
    def move(original: ti.i32, to: ti.i32):
        pass