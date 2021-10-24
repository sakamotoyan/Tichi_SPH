import taichi as ti
from sph_config import *

@ti.data_oriented
class PartNeighbs:
    def __init__(self,part_num,number_of_uids):
        self.neighbs=ti.field(int,shape=(part_num,max_neighb_per_part))
        self.neighbs_shift=ti.field(int,shape=(part_num,number_of_uids))
        self.neighbs_count=ti.field(int,shape=(part_num,number_of_uids))

@ti.data_oriented
class NeighborSearch:
    def __init__(self,number_of_uids):
        self.number_of_uids=number_of_uids
        self.node_part_count = ti.field(int)
        self.node_part_shift = ti.field(int)
        self.node_part_shift_count = ti.field(int)
        self.part_pid_in_node = ti.field(int)

        ti.root.dense(ti.i, node_num).dense(ti.j,number_of_uids).place(self.node_part_count) #i*j
        ti.root.dense(ti.i, node_num).dense(ti.j,number_of_uids).place(self.node_part_shift)
        ti.root.dense(ti.i, node_num).dense(ti.j,number_of_uids).place(self.node_part_shift_count)
        ti.root.dense(ti.i, max_part_num).place(self.part_pid_in_node)

        self.parts=[PartNeighbs(max_part_num_list[i],number_of_uids) for i in range(number_of_uids)]
        # self.neighbs=ti.field(int,shape=(max_part_num,max_neighb_per_part))
        # self.neighbs_shift=ti.field(int,shape=(max_part_num,number_of_uids))
        # self.neighbs_count=ti.field(int,shape=(max_part_num,number_of_uids))
        # self.part_shift=ti.field(int,shape=number_of_uids)

    @ti.kernel
    def clear_node(self):
        for i in range(node_num):
            for j in range(self.number_of_uids):
                self.node_part_count[i,j] = 0

    # #determine part_shift
    # def shift_part(self, *args):
    #     for obj in args:
    #         self.part_shift[obj.uid]=obj.part_num[None]
    #     sum=0
    #     for i in range(self.number_of_uids):
    #         self.part_shift[i],sum=sum,self.part_shift[i]+sum

    @ti.func
    def node_encode(self,pos: ti.template()):
        return int((pos - sim_space_lb[None])//sph_h[1])

    @ti.func
    def dim_encode(self,dim: ti.template()):
        return node_dim_coder[None].dot(dim)

    # determine node_part_count (particle count in each node for each uid)
    @ti.kernel
    def encode(self, obj: ti.template()):
        for i in range(obj.part_num[None]):
            node=self.node_encode(obj.pos[i])
            node_code = self.dim_encode(node)
            if 0 < node_code < node_num:
                ti.atomic_add(self.node_part_count[node_code,obj.uid], 1)

    # determine memory shift
    @ti.kernel
    def mem_shift(self):
        sum = ti.Vector([0])
        for i in range(node_num):
            for j in range(self.number_of_uids):
                self.node_part_shift[i,j] = ti.atomic_add(
                    sum[0], self.node_part_count[i,j])
                self.node_part_shift_count[i,j] = self.node_part_shift[i,j]

    # put obj particles into nodes
    @ti.kernel
    def fill_node(self, obj: ti.template()):
        for i in range(obj.part_num[None]):
            node=self.node_encode(obj.pos[i])
            node_code = self.dim_encode(node)
            if 0 < node_code < node_num:
                node_code_seq= ti.atomic_add(
                    self.node_part_shift_count[node_code,obj.uid], 1)
                self.part_pid_in_node[node_code_seq] = i

    # fill neighbs
    @ti.kernel
    def fill_neighbs(self, obj: ti.template()):
        # part=ti.static(self.parts[obj.uid])
        # p_shift=self.part_shift[obj.uid]
        for i in range(obj.part_num[None]):
            node=self.node_encode(obj.pos[i])
            count=0
            for k in range(self.number_of_uids):
                self.parts[obj.uid].neighbs_shift[i,k]=count
                for t in range(neighb_template.shape[0]):
                    node_code = self.dim_encode(node+neighb_template[t])
                    if 0 < node_code < node_num:
                        for j in range(self.node_part_count[node_code,k]):
                            shift = self.node_part_shift[node_code,k]+j
                            neighb_pid = self.part_pid_in_node[shift] 
                            self.parts[obj.uid].neighbs[i,count]=neighb_pid
                            count += 1
                self.parts[obj.uid].neighbs_count[i,k]=count-self.parts[obj.uid].neighbs_shift[i,k]

    # @ti.func
    # def neighbor(self,obj,nobj,i,j):
    #     return self.neighbs[i+self.part_shift[obj.uid],j+self.neighbs_shift[i+self.part_shift[obj.uid],nobj.uid]]

    # @ti.func
    # def neighbor_count(self,obj,nobj,i):
    #     return self.neighbs_count[i+self.part_shift[obj.uid],nobj.uid]

    # @ti.func
    # def neighbor1(self,obj,nobj,i,j):
    #     return self.neighbs[i+self.part_shift[obj.uid],j+]

    @ti.func
    def neighbor_first(self,obj,nobj,i):
        return self.parts[obj.uid].neighbs_shift[i,nobj.uid]
    
    @ti.func
    def neighbor_last(self,obj,nobj,i):
        return self.parts[obj.uid].neighbs_shift[i,nobj.uid] + self.parts[obj.uid].neighbs_count[i,nobj.uid]
            
    def establish_neighbs(self, *args):
        self.clear_node()
        # self.shift_part(*args)
        for obj in args:
            self.encode(obj)
        self.mem_shift()
        for obj in args:
            self.fill_node(obj)
        for obj in args:
            self.fill_neighbs(obj)
        
