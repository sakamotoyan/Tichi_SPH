from plyfile import PlyData, PlyElement

class Ply_writer:
    # file_name 为字符串，例如'pos.ply'这样的
    # save_path 为字符串，注意输入时末尾有没有 '\\' 或 '/' 都应可以，
    def __init__(self, file_name, save_path):
        self.attr_arr_stack_top = 0
        self.attr_arr_length = 0
        pass
    
    # attr_name 应为字符串，如'dt' 'time_step' 'part_num'等
    def push_single_attr(self, attr_name, attr_val):
        pass
    
    # 首个被写入的 attr_arr 必须是 'pos' 信息
    # 当 self.attr_arr_stack_top==0 时， attr_name 必须为为 'pos'，且做特殊处理
    # 当 self.attr_arr_stack_top==0 时， attr_arr 应为形如 [[1.1,2,3],[2,3,4], ... ] (dim==3) numpy float 数组，
    #   或[[1,2], [2,3], ...] (dim==2) numpy float 数组
    # 'pos' 需要特殊处理
    def push_attr_arr(self, attr_name, attr_arr) -> int:
        if self.attr_arr_stack_top == 0:
            if not attr_name=='pos':
                print('EXCEPTION from func push_attr_arr()')
                print('the first attribute of an array for a Ply_writer object must be \'pos\'')
                exit(0)
        attr_ID = self.attr_arr_stack_top
        self.attr_arr_length = attr_arr.shape[0]
        self.attr_arr_stack_top += 1
        return attr_ID

    # 实现文件写入
    # 如产生问题，需反馈产生了什么问题
    def flush(self):
        pass