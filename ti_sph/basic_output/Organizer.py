import taichi as ti
import numpy as np

from typing import List
from enum import Enum   

from ..basic_obj.Obj import Obj

@ti.data_oriented
class Organizer:

    class type(Enum):
        SEQ  = 0 # Organize all data in a sequential way.
        GRID = 1 # Organize all data in a grid way. This requires Obj to have {obj_name.node_index=ti.field(ti.f32, shape=(grid_node_num, dim))}.

    def __init__(self, format_type:type, obj:Obj) -> None:
        if not isinstance(format_type, self.type):
            raise ValueError(f"Invalid format type: {format_type}")
        if type is self.type.GRID and not hasattr(obj, "node_index"):
            raise ValueError(f"Obj {obj.__class__.__name__} does not have node_index field.")

        self.format_type = format_type
        self.obj = obj

        self.data_name_list:List[str] = []
        self.data_channel_list:List[int] = []

        # TODO: Consider 3D case.
        if self.format_type is self.type.GRID:
            self.np_node_index = self.obj.node_index.to_numpy()
            rows = self.np_node_index[:,0].max() + 1
            cols = self.np_node_index[:,1].max() + 1
            self.np_node_index_organized = np.empty((rows, cols, 2), dtype=int)
            self.np_data_organized = np.empty((rows, cols), dtype=float)

    def add_data(self, name:str, channel:int = 1):
        if not hasattr(self.obj, name):
            raise ValueError(f"{name} is not in {self.obj.__class__.__name__}.")
        if channel > self.obj.__dict__[name].n or channel < 1:
            raise ValueError(f"Channel {channel} is out of range of {name}({self.obj.__dict__[name].n} channels in total).")
        self.data_name_list.append(name)
        self.data_channel_list.append(channel)
    
    # TODO: Consider 3D case.
    def reshape_data(self, np_data):
        A = self.np_node_index
        B = np_data
        A_organized = self.np_node_index_organized
        B_organized = self.np_data_organized
        A_organized[A[:, 0], A[:, 1]] = A
        B_organized[A[:, 0], A[:, 1]] = B

        A_organized = np.flip(A_organized, axis=0)
        B_organized = np.flip(B_organized, axis=0)
        
        return B_organized
            

    def export_to_numpy(self, index:int=0, path:str=".", truncate:bool = False, truncate_num:int = 0):
        
        for data_name in self.data_name_list:
            data = getattr(self.obj, data_name)
            file_name = f"{path}/{data_name}_{index}"

            for channel in range(self.data_channel_list[self.data_name_list.index(data_name)]):
                file_name_channel = f"{file_name}_{channel}"
                np_data = data.to_numpy()[:, channel]
                if self.format_type is self.type.SEQ and not truncate:
                    np.save(file_name_channel, np_data)
                elif self.format_type is self.type.SEQ and truncate:
                    np.save(file_name_channel, np_data[:truncate_num])
                elif self.format_type is self.type.GRID:
                    np_data_reshape = self.reshape_data(np_data)