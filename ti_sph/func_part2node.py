import taichi as ti
import math

# NOTE: $span$ should be float or int
# OUTPUT: ti.Vector(dim, int)
@ti.func
def part2node_pos_aligner(pos: ti.template(), node_coordinate_base: ti.template(), node_coordinate_span):
    pos_aligner = int(ti.floor((pos - node_coordinate_base) / node_coordinate_span))
    return pos_aligner

@ti.func
def part2node_array_positioner(pos_aligner: ti.template(), node_structure_coder: ti.template()):
    array_positioner = pos_aligner.dot(node_structure_coder)
    return array_positioner