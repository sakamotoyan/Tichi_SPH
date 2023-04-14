import taichi as ti
from ..basic_op.type import *
from ..basic_solver_funcs.SPH_funcs import *

'''#################### BELOW IS THE TEMPLATE FOR NEIGHBORHOOD SEASCHING ####################'''
@ti.data_oriented
class Neighb_search_template:
    def __init__(
        self,
        dim,  # int32
        search_cell_range,  # int32
    ):
        self.search_cell_range = val_i(0)
        self.search_cell_range[None] = search_cell_range

        self.search_template = ti.Vector.field(
            dim, ti.i32, (self.search_cell_range[None] * 2 + 1) ** dim
        )

        self.neighb_dice = ti.field(
            ti.i32, self.search_cell_range[None] * 2 + 1)

        for i in range(self.neighb_dice.shape[0]):
            self.neighb_dice[i] = -self.search_cell_range[None] + i

        for i in range(self.search_template.shape[0]):
            tmp = i
            for j in ti.static(range(dim)):
                digit = tmp // (self.neighb_dice.shape[0] ** (dim - j - 1))
                tmp = tmp % (self.neighb_dice.shape[0] ** (dim - j - 1))
                self.search_template[i][dim - j - 1] = self.neighb_dice[digit]

    @ti.func
    def get_neighb_cell_num(self):
        return self.search_template.shape[0]
    
    @ti.func
    def get_neighb_cell_vec(self, i):
        return self.search_template[i]
'''#################### ABOVE IS THE TEMPLATE FOR NEIGHBORHOOD SEASCHING ####################'''






'''#################### BELOW IS THE CACHE STRUCT AND ACCOMPINED POINTER 
                        STRUCT FOR LOGGING NEIGHBOUR PARTICLES ####################'''
@ti.dataclass
class Pointer_pool:
    begin: ti.i32
    current: ti.i32
    size: ti.i32

@ti.dataclass
class Pointer_obj:
    begin: ti.i32
    size: ti.i32

@ti.dataclass
class LinkedList_container:
    neighb_part_id: ti.i32
    neighb_obj_id: ti.i32
    next: ti.i32
'''#################### ABOVE IS THE CACHE STRUCT AND ACCOMPINED POINTER
                        STRUCT FOR LOGGING NEIGHBOUR PARTICLES ####################'''  






'''#################### BELOW IS THE CLASS FOR NEIGHBORHOOD SEASCHING ####################'''
@ti.data_oriented
class Neighb_list:
    ''' init the neighb list'''
    def __init__(
            self,
            obj__: ti.template(),  # Particle class
            obj_pos_: ti.template(),  # ti.Vector.field(dim, ti.f32, obj_part_num)
            obj_stack_top_: ti.template(),  # vali_()
            max_neighb_part_num_: ti.template(),  # vali_()
            max_neighb_obj_num_: ti.template(), # vali_()
    ):
        self.obj__ = obj__
        self.obj_pos_ = obj_pos_
        self.obj_part_num_ = val_i(self.obj_pos_.shape[0])
        self.obj_stack_top_ = val_i(obj_stack_top_[None])
        self.max_neighb_part_num_ = val_i(max_neighb_part_num_[None])
        self.max_neighb_obj_num_ = val_i(max_neighb_obj_num_[None])
        self.dim = ti.static(self.obj_pos_.n)

        self.neighb_obj_list = []  # Particle class
        self.neighb_obj_pos_list = []  # ti.Vector.field(dim, ti.f32, neighb_obj_part_num)
        self.neighb_cell_list = []  # Neighb_cell_simple class
        self.neighb_search_range_list = []  # val_f() # TODO: 接收 Dynamic 作为搜索范围
        self.neighb_search_template_list = []  # Neighb_search_template class

        '''[DIY AREA]'''
        ''' add your own data here'''
        Struct_cached_attributes = ti.types.struct(
            dist=ti.f32,
            xij_norm=vecxf(self.dim),
            W=ti.f32,
            grad_W=vecxf(self.dim),
        )

        self.neighb_pool_used_space_ = val_i(0)
        self.neighb_pool_size_ = ti.static(self.max_neighb_part_num_)
        self.neighb_pool_pointer_ = Pointer_pool.field(shape=(self.obj_part_num_[None]))
        self.neighb_obj_pointer_ = Pointer_obj.field(shape=(self.obj_part_num_[None], self.max_neighb_obj_num_[None]))
        self.neighb_pool_container_ = LinkedList_container.field(shape=(max_neighb_part_num_[None]))
        self.cached_neighb_attributes_ = Struct_cached_attributes.field(shape=(max_neighb_part_num_[None]))

    ''' clear the cache pool'''
    @ti.kernel
    def clear_pool(self):
        for part_id in range(self.obj_stack_top_[None]):
            self.neighb_pool_pointer_[part_id].begin = -1
            self.neighb_pool_pointer_[part_id].current = -1
            self.neighb_pool_pointer_[part_id].size = 0
            for obj_seq in range(self.max_neighb_obj_num_[None]):
                self.neighb_obj_pointer_[part_id, obj_seq].begin = -1
                self.neighb_obj_pointer_[part_id, obj_seq].size = 0
        self.neighb_pool_used_space_[None] = 0

    ''' add a $neighb obj$ to the neighb search range'''
    def add_neighb_obj(
            self,
            neighb_obj__: ti.template(),  # Particle class
            neighb_obj_pos_: ti.template(),  # ti.Vector.field(dim, ti.f32, neighb_obj_part_num)
            neighb_cell__: ti.template(),  # Neighb_cell_simple class
            search_range_: ti.template(),  # val_f() # TODO: 接收 Dynamic 作为搜索范围
    ):
        ''' check input validity'''
        if neighb_obj__ in self.neighb_obj_list:
            raise Exception("neighb_obj already in list")
        if neighb_cell__ in self.neighb_cell_list:
            raise Exception("neighb_cell already in list")
        if not hasattr(neighb_obj__, "id"):
            raise Exception("neighb_obj has no id")
        if self.dim != neighb_obj_pos_.n:
            raise Exception("dim not equal")

        ''' append to lists '''
        self.neighb_obj_list.append(neighb_obj__)
        self.neighb_obj_pos_list.append(neighb_obj_pos_)
        self.neighb_cell_list.append(neighb_cell__)
        self.neighb_search_range_list.append(search_range_)

        ''' generate search template '''
        search_cell_range = ti.ceil(search_range_[None] / neighb_cell__.cell_size_[None])
        neighb_search_template__ = Neighb_search_template(self.dim, search_cell_range)
        self.neighb_search_template_list.append(neighb_search_template__)

    ''' get a obj, neighb_obj attributes pair  one at a time, as inputs to register_a_neighbour() '''
    def register_neighbours(
        self,
    ):
        self.clear_pool()
        for i in range(len(self.neighb_obj_list)):
            self.register_a_neighbour(self.neighb_obj_list[i].id, self.neighb_search_range_list[i][None], self.neighb_obj_pos_list[i], self.neighb_cell_list[i], self.neighb_search_template_list[i])
        if not self.neighb_pool_size_[None] > self.neighb_pool_used_space_[None]:
            raise Exception("neighb_pool overflow")
        # print("debug: neighb_pool_used_space_ = ", self.neighb_pool_used_space_[None], " / ", self.neighb_pool_size_[None], " = ", self.neighb_pool_used_space_[None] / self.neighb_pool_size_[None]*100, " %")
    ''' register all particles form a $neighbour obj$ to $obj particles$ as neighbours '''
    @ti.kernel
    def register_a_neighbour(
        self,
        neighb_obj_id: ti.i32,
        search_range: ti.f32,
        neighb_pos_: ti.template(),  # ti.Vector.field(dim, ti.f32, neighb_obj_part_num)
        neighb_cell__: ti.template(),  # Neighb_cell_simple class
        neighb_search_template__: ti.template(),  # Neighb_search_template class
    ):
        for part_id in range(self.obj_stack_top_[None]):
            size_before = self.neighb_pool_pointer_[part_id].size

            ''' locate the cell where the $obj particle$ is located '''
            located_cell_vec = neighb_cell__.compute_cell_vec(self.obj_pos_[part_id])
            ''' iterate over all neighbouring cells '''
            for neighb_cell_iter in range(neighb_search_template__.get_neighb_cell_num()):
                ''' get the $cell vector$ of the neighbouring cell through the template'''
                neighb_cell_vec = located_cell_vec + neighb_search_template__.get_neighb_cell_vec(neighb_cell_iter)
                ''' check if the neighbouring cell is within the domain '''
                if not neighb_cell__.within_cell(neighb_cell_vec):
                    continue
                ''' get the neighbouring cell id by encoding the $cell vector$ '''
                neighb_cell_id = neighb_cell__.encode_cell_vec(neighb_cell_vec)
                ''' get the number of particles in the neighbouring cell '''
                part_num = neighb_cell__.get_part_num_in_cell(neighb_cell_id)
                for part_iter in range(part_num):
                    ''' get the particle id in the neighbouring cell '''
                    neighb_part_id = neighb_cell__.get_part_id_in_cell(neighb_cell_id, part_iter)
                    dist = (self.obj_pos_[part_id] - neighb_pos_[neighb_part_id]).norm()
                    ''' register the neighbouring particle '''
                    if dist < search_range:
                        pointer = self.insert_a_part(part_id, neighb_obj_id, neighb_part_id, dist)
                        
                        ''' [DIY AREA] '''
                        ''' You can add attributes you want to be pre-computed here '''
                        self.cached_neighb_attributes_[pointer].dist = dist
                        self.cached_neighb_attributes_[pointer].xij_norm = (self.obj_pos_[part_id] - neighb_pos_[neighb_part_id]) / dist
                        self.cached_neighb_attributes_[pointer].W = spline_W(dist, self.obj__.sph_[part_id].h, self.obj__.sph_[part_id].sig)
                        self.cached_neighb_attributes_[pointer].grad_W = grad_spline_W(dist, self.obj__.sph_[part_id].h, self.obj__.sph_[part_id].sig_inv_h) * self.cached_neighb_attributes_[pointer].xij_norm

            self.neighb_obj_pointer_[part_id, neighb_obj_id].size = self.neighb_pool_pointer_[part_id].size - size_before

    ''' insert a neighbouring particle into the linked list'''
    @ti.func
    def insert_a_part(
        self,
        part_id: ti.i32,
        neighb_obj_id: ti.i32,
        neighb_part_id: ti.i32,
        dist: ti.f32,
    ) -> ti.i32:
        pointer = ti.atomic_add(self.neighb_pool_used_space_[None], 1)
        self.neighb_pool_pointer_[part_id].size += 1
        
        if self.neighb_pool_pointer_[part_id].begin == -1:
            self.neighb_pool_pointer_[part_id].begin = pointer
        else:
            self.neighb_pool_container_[self.neighb_pool_pointer_[part_id].current].next = pointer
        
        if self.neighb_obj_pointer_[part_id, neighb_obj_id].begin == -1:
            self.neighb_obj_pointer_[part_id, neighb_obj_id].begin = pointer

        self.neighb_pool_pointer_[part_id].current = pointer
        self.neighb_pool_container_[pointer].neighb_obj_id = neighb_obj_id
        self.neighb_pool_container_[pointer].neighb_part_id = neighb_part_id

        return pointer
'''#################### ABOVE IS THE CLASS FOR NEIGHBORHOOD SEASCHING ####################'''

