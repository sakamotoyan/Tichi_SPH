import taichi as ti

@ti.data_oriented
class Neighb_looper:
    def __init__(self) -> None:
        pass

    @ti.kernel
    def loop(self, neighb_pool:ti.template(), neighb_obj:ti.template(), func:ti.template()):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            neighb_part_num = neighb_pool.neighb_obj_pointer[part_id, neighb_obj.ti_get_id()[None]].size
            neighb_part_shift = neighb_pool.neighb_obj_pointer[part_id, neighb_obj.ti_get_id()[None]].begin
            for neighb_part_iter in range(neighb_part_num):
                neighb_part_id = neighb_pool.neighb_pool_container[neighb_part_shift].neighb_part_id
                ''' Code for Computation'''
                func(part_id, neighb_part_id, neighb_part_shift, neighb_pool, neighb_obj)
                ''' End of Code for Computation'''
                ''' DO NOT FORGET TO COPY/PASE THE FOLLOWING CODE WHEN REUSING THIS FUNCTION '''
                neighb_part_shift = neighb_pool.neighb_pool_container[neighb_part_shift].next