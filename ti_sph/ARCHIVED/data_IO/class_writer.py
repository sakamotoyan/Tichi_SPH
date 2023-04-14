from plyfile import PlyData, PlyElement
import numpy as np
import os


class Ply_writer:
    # file_name 为字符串，例如'pos.ply'这样的
    # save_path 为字符串，注意输入时末尾有没有 '\\' 或 '/' 都应可以，
    def __init__(self, file_name, save_path):
        self.file_name = file_name
        self.save_path = save_path
        self.single_attr = []
        self.single_attr_dtype = []
        self.attr_arr_stack_top = 0
        self.attr_arr_length = 0
        self.elements = []

    # attr_name 应为字符串，如'dt' 'time_step' 'part_num'等
    def add_single_attr(self, attr_name, attr_val, dtype):
        self.single_attr_dtype.append((attr_name, dtype))
        self.single_attr.append(attr_val)

    def push_single_attr(self, ele_name):
        save_signal = np.array([tuple(self.single_attr)], dtype=self.single_attr_dtype)
        self.elements.append(PlyElement.describe(save_signal, ele_name))

        self.single_attr_dtype.clear()
        self.single_attr.clear()

    # 首个被写入的 attr_arr 必须是 'pos' 信息
    # 当 self.attr_arr_stack_top==0 时， attr_name 必须为为 'pos'，且做特殊处理
    # 当 self.attr_arr_stack_top==0 时， attr_arr 应为形如 [[1.1,2,3],[2,3,4], ... ] (dim==3) numpy float 数组，
    #   或[[1,2], [2,3], ...] (dim==2) numpy float 数组
    # 'pos' 需要特殊处理
    def push_attr_arr(self, attr_name, attr_arr, data_type) -> int:
        if self.attr_arr_stack_top == 0:
            if not attr_name == "pos":
                print(
                    "EXCEPTION from func push_attr_arr(): the first attribute of an array for a Ply_writer object must be 'pos'"
                )
                exit(0)

        save_data = np.array([tuple(item) for item in attr_arr], dtype=data_type)
        self.elements.append(PlyElement.describe(save_data, attr_name))

        attr_ID = self.attr_arr_stack_top
        self.attr_arr_length = attr_arr.shape[0]
        self.attr_arr_stack_top += 1
        return attr_ID

    # 实现文件写入
    # 如产生问题，需反馈产生了什么问题
    def flush(self):
        if not os.path.exists(self.save_path):
            print(
                "EXCEPTION from func flush(): the directory '"
                + self.save_path
                + "' does not exist"
            )
            exit(0)

        PlyData(self.elements, text=True).write(
            os.path.join(self.save_path, self.file_name)
        )
        self.elements.clear()


def ply_write_test(file_name, save_path):
    writer = Ply_writer(file_name, save_path)
    writer.add_single_attr("dt", 0.001, "f4")
    writer.add_single_attr("part_num", 4000, "i4")
    writer.add_single_attr("time_step", 0.5, "f4")
    writer.push_single_attr("simulation_info")

    a = np.array([[1, 2], [2, 2], [3, 3]])
    b = np.array([[2, 2, 3], [2, 2, 3], [3, 3, 3]])
    dim = 2
    phase_num = 3
    pos_dype = []
    volume_frac_dtype = []
    p = ['x', 'y', 'z']
    for i in range(dim):
        pos_dype.append((p[i], 'f4'))
    for k in range(phase_num):
        volume_frac_dtype.append(('f' + str(k + 1), 'f4'))
    writer.push_attr_arr('pos', a, pos_dype)
    writer.push_attr_arr('volume_frac', b, volume_frac_dtype)

    writer.flush()


