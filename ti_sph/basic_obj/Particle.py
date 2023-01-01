import taichi as ti


@ti.data_oriented
class Particle:
    def __init__(
        self,
        part_num,
    ):
        if(part_num <= 0):
            raise ValueError("part_num must be larger than 0")

        self.part_num = part_num
        self.stack_top = 0

        self.attr_list = {}
        self.array_list = {}
        self.struct_list = {}

        # seq_log to track the order of particles (and the swap of particles)
        self.seq_log = ti.field(ti.i32, (part_num))
        for i in range(part_num):
            self.seq_log[i] = i

    # Functions Type 1: structure management
    def add_attr(self, name, attr):
        if name in self.__dict__:
            print(
                f"Warning: {name} already exists in {self.__class__.__name__}")
        else:
            if(isinstance(attr, list)):
                self.__dict__[name] = []
                self.attr_list[name] = []
                for attr_i in attr:   
                    self.__dict__[name].append(attr_i)
            
            else:
                self.__dict__[name] = attr
            
            self.attr_list[name] = self.__dict__[name]

    # This function is deprecated (has been merged with add_attr()) and will be removed in the future.
    def add_attrs(self, name, attrs):
        if name in self.__dict__:
            print(
                f"Warning: {name} already exists in {self.__class__.__name__}")
        else:
            self.__dict__[name] = []
            self.attr_list[name] = []
            for attr in attrs:   
                self.__dict__[name].append(attr)
            self.attr_list[name] = self.__dict__[name]

    def add_array(self, name, array):
        if name in self.__dict__:
            print(
                f"Warning: {name} already exists in {self.__class__.__name__}")
        else:
            if(isinstance(array,list)):
                self.__dict__[name] = []
                self.array_list[name] = []
                for array_i in array:   
                    ti.root.dense(ti.i, self.part_num).place(array_i)
                    self.__dict__[name].append(array_i)
            
            else:
                ti.root.dense(ti.i, self.part_num).place(array)
                self.__dict__[name] = array
            
            self.array_list[name] = self.__dict__[name]

    # This function is deprecated (has been merged with add_array()) and will be removed in the future.
    def add_arrays(self, name, arrays): 
        if name in self.__dict__:
            print(
                f"Warning: {name} already exists in {self.__class__.__name__}")
        else:
            self.__dict__[name] = []
            self.array_list[name] = []
            for array in arrays:   
                ti.root.dense(ti.i, self.part_num).place(array)
                self.__dict__[name].append(array)
            self.array_list[name] = self.__dict__[name]

    def add_struct(self, name, struct):
        if name in self.__dict__:
            print(
                f"Warning: {name} already exists in {self.__class__.__name__}")
        else:
            if(isinstance(struct,list)):
                self.__dict__[name] = []
                self.struct_list[name] = []
                for struct_i in struct:   
                    self.__dict__[name].append(struct_i.field(shape=(self.part_num,)))
            else:
                self.__dict__[name] = struct.field(shape=(self.part_num,))
            self.struct_list[name] = self.__dict__[name]

    def add_structs(self, name, structs):
        if name in self.__dict__:
            print(
                f"Warning: {name} already exists in {self.__class__.__name__}")
        
        self.__dict__[name] = []
        self.struct_list[name] = []
        for struct in structs:   
            self.__dict__[name].append(struct.field(shape=(self.part_num,)))
        self.struct_list[name] = self.__dict__[name]


    # Functions Type 2: verbose functions
    def verbose_structs(self, append=None):
        if append is not None:
            print("")
            text_color_begin = "\033[4;31;43m"
            text_color_end = "\033[0m"
            print(f"{text_color_begin}{append}{text_color_end}")
        sub_element_template = "  {0:16}\t|\t{1:8}\t|\t{2:16}"
        head_template = "{0:16}\t|"
        tilde_template = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        dash_template =  "---------------------------------------------------------------------------------------"
        print(tilde_template.format())
        print(f"{self.__class__.__name__} structs: ")
        print(tilde_template.format())
        for name in self.__dict__:
            if name in self.struct_list:
                if(isinstance(self.__dict__[name],list)):
                    iter = 0
                    for struct in self.__dict__[name]:
                        print(head_template.format(f"{name}[{iter}]"))
                        for key in struct.keys:
                            # print('self.' + name + '[' + str(iter) + ']' + '.' + key  + '.dtype')
                            print(sub_element_template.format(
                                key, 
                                f"dtype={eval('self.' + name + '[' + str(iter) + ']' + '.' + key  + '.dtype')}", 
                                f"{[eval('self.' + name + '[' + str(iter) + ']' + '.' + key)]}"))
                        iter += 1
                        print(dash_template.format())
                else:
                    print(f"{head_template.format(name)}")
                    for key in self.__dict__[name].keys:
                        print(sub_element_template.format(key, f"dtype={eval('self.'+name+'.'+key).dtype}", f"{[eval('self.'+name+'.'+key)]}"))
                    print(dash_template.format())
    
    def verbose_arrays(self, append=None):
        if append is not None:
            print("")
            text_color_begin = "\033[4;31;43m"
            text_color_end = "\033[0m"
            print(f"{text_color_begin}{append}{text_color_end}")
        element_template = "{0:16}\t|\t{1:8}\t|\t{2:16}"
        sub_element_template = "  {0:16}\t|\t{1:8}\t|\t{2:16}"
        head_template = "{0:16}\t|"
        tilde_template = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        dash_template =  "---------------------------------------------------------------------------------------"
        print(tilde_template.format())
        print(f"{self.__class__.__name__} arrays:")
        print(tilde_template.format())
        for name in self.__dict__:
            if name in self.array_list:
                if(isinstance(self.__dict__[name],list)):
                    print(head_template.format(name))
                    iter = 0
                    for array in self.__dict__[name]:
                        print(sub_element_template.format(f"[{iter}]", f"dtype={array.dtype}", f"{[array]}"))
                        iter += 1
                else:
                    print(element_template.format(name, f"dtype={self.__dict__[name].dtype}", f"{[self.__dict__[name]]}"))
                print(dash_template.format())
        print(" ")

    def verbose_attrs(self, append=None):
        if append is not None:
            print("")
            text_color_begin = "\033[4;31;43m"
            text_color_end = "\033[0m"
            print(f"{text_color_begin}{append}{text_color_end}")
        element_template = "{0:16}\t|\t{1:8}\t|\t{2:16}"
        sub_element_template = "  {0:16}\t|\t{1:8}\t|\t{2:16}"
        head_template = "{0:16}\t|"
        tilde_template = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        dash_template =  "---------------------------------------------------------------------------------------"
        print(tilde_template.format())
        print(f"{self.__class__.__name__} attrs:")
        print(tilde_template.format())
        for name in self.__dict__:
            if name in self.attr_list:
                if(isinstance(self.__dict__[name],list)):
                    print(head_template.format(name))
                    iter = 0
                    for attr in self.__dict__[name]:
                        print(sub_element_template.format(f"[{iter}]", f"type={type(attr)}", f"{attr}"))
                        iter += 1
                else:
                    print(element_template.format(name, f"type={type(self.__dict__[name])}", f"{self.__dict__[name]}"))
                print(dash_template.format())
        print(" ")

    # Functions Type 3: Data operations
    # @ti.kernel
    def from_numpy(self, to:ti.template(), data:ti.types.ndarray()):
        num = data.shape[0]
        print(f"num={num}")
        arr = to.to_numpy()
        arr[self.stack_top:num,:] = data
        to.from_numpy(arr)
    
    def update_stack_top(self, num):
        self.stack_top += num


