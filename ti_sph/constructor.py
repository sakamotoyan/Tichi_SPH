import taichi as ti

from .struct_node import *
from .struct_config import *


def node_construct(self, dim, node_num, capacity_list):
    for capacity in capacity_list:
        if capacity == "node_basic":
            self.basic = struct_node_basic(dim, node_num)
        if capacity == "node_color":
            self.color = struct_node_color(node_num)
        if capacity == "node_sph":
            self.sph = struct_node_sph(dim, node_num)
        if capacity == "node_implicit_sph":
            self.implicit_sph = struct_node_implicit_sph(dim, node_num)
        if capacity == "node_neighb_search":
            self.located_cell = struct_node_neighb_search(dim, node_num)
        if capacity == "node_ISPH_Elastic":
            self.elastic_sph = struct_node_elastic_sph(dim, node_num)


def node_neighb_cell_construct(self, dim, neighb_cell_num, capacity_list):
    for capacity in capacity_list:
        if capacity == "node_neighb_search":
            self.cell = struct_node_neighb_cell(neighb_cell_num)


def info_construct(self, dim, capacity_list):
    for capacity in capacity_list:
        if capacity == "info_space":
            self.space = struct_config_space(dim)
        if capacity == "info_discretization":
            self.discre = struct_config_discretization()
        if capacity == "info_sim":
            self.sim = struct_config_sim(dim)
        if capacity == "info_neighb_search":
            self.neighb = struct_config_neighb_search(dim)
        if capacity == "info_gui":
            self.gui = struct_config_gui()
