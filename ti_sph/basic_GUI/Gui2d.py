from typing import Tuple

import taichi as ti
from ..basic_op.type import *

class Gui2d:
    def __init__(self, res: Tuple[int, int]=(512, 512), title: str='Simulation'):
        gui = ti.GUI(res, title)