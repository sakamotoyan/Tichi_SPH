import taichi as ti

""" Type"""
vec2f = ti.types.vector(2, ti.f32)
vec2i = ti.types.vector(2, ti.i32)

vec3f = ti.types.vector(3, ti.f32)
vec3i = ti.types.vector(3, ti.i32)

vec4f = ti.types.vector(4, ti.f32)
vec4i = ti.types.vector(4, ti.i32)


def vecxf(n):
    return ti.types.vector(n, ti.f32)


def vecxi(n):
    return ti.types.vector(n, ti.i32)
