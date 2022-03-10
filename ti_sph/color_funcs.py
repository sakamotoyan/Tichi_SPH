import taichi as ti

# Input:  rgb-> vec->ti.Vector([{3,float}])
# Output: hex-> integer looks like 0xFFFFFF
@ti.func
def rgb_vec2hex(rgb: ti.template()):  # r, g, b are normalized
    return ((int(rgb[0] * 255)) << 16) + ((int(rgb[1] * 255)) << 8) + (int(rgb[2] * 255))

# Input:  hex-> integer looks like 0xFFFFFF
# Output: rgb-> vec->ti.Vector([{3,float}])
@ti.func
def rgb_hex2vec(hex: int):  # r, g, b are normalized
    return float(ti.Vector([(hex & 0xFF0000) >> 16, (hex & 0x00FF00) >> 8, (hex & 0x0000FF)])) / 255