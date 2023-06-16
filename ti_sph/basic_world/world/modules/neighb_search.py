import taichi as ti

def update_pos_in_neighb_search(self):
    for part_obj in self.part_obj_list:
        if not part_obj.m_is_dynamic:
            continue
        if part_obj.m_neighb_search is None:
            continue
        part_obj.m_neighb_search.update_self()

def update_neighbs(self):
    for part_obj in self.part_obj_list:
        if part_obj.m_neighb_search is None:
            continue
        part_obj.m_neighb_search.search_neighbors()

def search_neighb(self):
    self.update_pos_in_neighb_search()
    self.update_neighbs()