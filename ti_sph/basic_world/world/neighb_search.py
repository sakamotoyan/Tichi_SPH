import taichi as ti

def update_pos_in_neighb_search(self):
    for part_obj in self.part_obj_list:
        if not part_obj.is_active:
            continue
        if part_obj.m_neighb_search is None:
            continue
        part_obj.m_neighb_search.neighb_cell.update_self()