from plyfile import PlyData, PlyElement
import os

class Ply_reader:
    def __init__(self):
        self.file_path = None
        self.read_buffer = None
        self.single_attr_list = None
        self.attr_arr_list = {}

    def set_file_path(self, file_path):
        if not os.path.exists(file_path):
            print('EXCEPTION from func set_file_path(): the file \''+file_path+'\' does not exist')
            exit(0)
        self.file_path = file_path

    def read(self):
        if not self.file_path:
            print('EXCEPTION from func read(): the \'file_path\' is None')
            exit(0)
        self.read_buffer = PlyData.read(self.file_path)

    def get_single_attr_list(self, ele_name):
        if not self.read_buffer:
            print('EXCEPTION from func get_single_attr_list(): the \'read_buffer\' is None')
            os._exit(0)
        try:
            self.single_attr_list = self.read_buffer[ele_name].data
            return self.single_attr_list
        except:
            print('EXCEPTION from func get_single_attr_list(): the element \''+ele_name+'\' does not exist')
            exit(0)

    def get_attr_arr_list(self):
        if not self.read_buffer:
            print('EXCEPTION from func get_single_attr_list(): the \'read_buffer\' is None')
            os._exit(0)
        for ele in self.read_buffer:
            if len(ele) > 1:
                self.attr_arr_list.update({ele.name: ele.data})
        return self.attr_arr_list

    def get_single_attr(self, attr_name, ele_name=None):
        try:
            if ele_name is not None:
                single_attr = self.get_single_attr_list(ele_name)
                return single_attr[attr_name][0]
            else:
                if not self.single_attr_list:
                    print('EXCEPTION from func get_attr_arr_list(): single_attr_list is empty')
                    os._exit(0)
                else:
                    print("single_attr_list")
                    return self.single_attr_list[attr_name][0]
        except:
            if not ele_name:
                print('EXCEPTION from func get_attr_arr_list(): '
                      'the attribute \'' + attr_name + '\' does not exist')
            else:
                print('EXCEPTION from func get_attr_arr_list(): '
                      'the element \'' + ele_name + '\'  or the attribute \''+attr_name+'\' does not exist')
            exit(0)

    def get_attr_arr(self, attr_name):
        try:
            return self.read_buffer[attr_name].data
        except:
            print('EXCEPTION from func get_attr_arr(): the attribute \'' + attr_name + '\' does not exist')
            exit(0)


def ply_reader_test(file_path):
    reader = Ply_reader()
    reader.set_file_path(file_path)
    reader.read()

    single_attr = reader.get_single_attr_list('simulation_info')
    print(single_attr['part_num'][0])

    dt = reader.get_single_attr('dt', 'simulation_info')
    print(dt)

    attr_arr = reader.get_attr_arr_list()
    print(attr_arr['pos'])

    volume_frac = reader.get_attr_arr('volume_frac')
    print(volume_frac)




