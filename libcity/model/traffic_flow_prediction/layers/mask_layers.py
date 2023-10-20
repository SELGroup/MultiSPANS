import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import scipy.sparse as sp
from infomap import Infomap
import networkx as nx
import math
from torch_geometric.utils import to_dense_adj

class Mask_Bias_Generator():
    """
        mask_heads_share : [True: shape (q,k,h)
                            False:  shape (q,k) or (t/n,q,k)]
        
    """
    def __init__(self, q_size, v_size):
        self.q_size = q_size
        self.v_size = v_size
        self._bias = None
        self._mask = None

    def get(self):
        return self._bias or self._mask




class Graph_Mask_Generator(Mask_Bias_Generator):
    """
        mask_heads_share: True, single graph, False, Multi-relation graph
        graph
    """
    def __init__(self, num_node, graph_data):
        super(Graph_Mask_Generator,self).__init__(num_node, num_node)
        self.num_node = num_node
        if len(graph_data) == len(graph_data[0]) and type(graph_data)==np.ndarray: # dense
            dense_adj_mx = graph_data
        else: # edge index
            dense_adj_mx = to_dense_adj(edge_index = graph_data, max_num_nodes=self.num_node)
        assert (
            dense_adj_mx.shape[0]==self.q_size and dense_adj_mx.shape[1]==self.v_size
        ), "Wrong adj matrix"
        dense_adj_mx = torch.from_numpy(graph_data)
        out0 = dense_adj_mx>0
        out1 = torch.where(dense_adj_mx==torch.inf,
                        torch.tensor([False,]).expand(dense_adj_mx.shape),
                        torch.tensor([True,]).expand(dense_adj_mx.shape))
        self._mask = out0 * out1





class Infomap_Multi_Mask_Generator(Mask_Bias_Generator):
    # get masks shaped as [q x k x h]
    def __init__(self, num_node, graph_data):
        super(Infomap_Multi_Mask_Generator,self).__init__(num_node, num_node)
        self.im = Infomap(silent=True,num_trials=20)
        self.num_node = num_node
        if type(graph_data) is (nx.DiGraph or nx.Graph):
            self.G = graph_data
        else:
            self.G = nx.DiGraph(graph_data) # dense_adj_mx
        self.im.add_networkx_graph(self.G)
        self._gen_mask()
    
    def _gen_mask(self):
        self.im.run()
        im = self.im
        num_levels = im.num_levels
        max_num_module = im.num_leaf_modules
        self.num_mask = num_levels-1
        masks = list()
        for each_level in range(1,num_levels):
            itr = im.get_nodes(depth_level=each_level)
            # clu_tag = torch.zeros([self.num_node,],dtype=torch.int)
            # clu_tags = torch.zeros([max_num_module,self.num_node],dtype=torch.int)
            clu_tags = torch.full([max_num_module,self.num_node],-1,dtype=torch.int)
            # ind = torch.zeros([self.num_node,],dtype=torch.int)
            for each in itr:
                # 一个行复制，一个列复制，用两个矩阵where相等为True,overlap部分无法处理(mend)
                clu_tags[each.module_id-1][each.node_id] = 1
            temp1 = clu_tags.unsqueeze(2).expand([max_num_module,self.num_node,self.num_node])
            temp2 = temp1.transpose(1,2)
            out = torch.any((temp1==temp2)*(temp1!=-1),dim=0)
            masks.append(out)
        masks = torch.stack(masks,dim=2)
        self._mask = masks





class Infomap_Multilevel_Bias_Generator(Mask_Bias_Generator):
    """
        Input: static graph_data (nx.Digraph or dense_adj_mx)
        Output: bias , shape=(node_num,node_num), dtype=float
    """ 
    def __init__(self, num_node, graph_data, bias_scale_type=0):
        super(Infomap_Multilevel_Bias_Generator,self).__init__(True, num_node, num_node)
        self.im = Infomap(silent=True,num_trials=20)
        self.bias_scale_type = bias_scale_type
        # only for static graph
        if type(graph_data) is (nx.DiGraph or nx.Graph):
            self.G = graph_data
        else:
            self.G = nx.DiGraph(graph_data) # dense_adj_mx
        self.im.add_networkx_graph(self.G)
        self._gen_bias()
    
    def _gen_bias(self):
        self.im.run()
        # read tree
        bias_type = self.bias_scale_type
        im = self.im
        num_nodes=im.num_nodes
        itr = im.get_tree(depth_level=1, states=True)
        path_modcentral_dict=dict()
        path_nodeid_dict = dict()
        for each in itr:
            path_modcentral_dict[each.path] = each.modular_centrality
            if each.is_leaf:
                path_nodeid_dict[each.path] = each.node_id
        
        single_layer_att_bias = torch.zeros([num_nodes,num_nodes],dtype=torch.float64)

        nodes = im.get_nodes(depth_level=1, states=True)
        for each in nodes:
            path = each.path
            nd_from = torch.Tensor([path_nodeid_dict[path],]).type(torch.long)
            for i in range(len(path),0,-1): # 0 for min attention layer
                now_path = path[:i]
                common_prefix = now_path[:-1]
                ## bias type
                if bias_type == 0:
                    b1 = path_modcentral_dict[now_path]
                elif bias_type == 1:
                    b1 = math.exp(path_modcentral_dict[now_path])
                else:
                    b1 = 1
                
                nd_to = []
                for key in path_nodeid_dict.keys():
                    if key[:len(common_prefix)]==common_prefix: # key!=now_path
                        nd_to.append(path_nodeid_dict[key])
                nd_to = torch.Tensor(nd_to).type(torch.long)
                single_layer_att_bias[nd_from,nd_to]+=b1
                single_layer_att_bias[nd_to,nd_from]+=b1
        self._bias = single_layer_att_bias/2

def get_static_multihead_mask(num_head,mask_generator_list:list,device=torch.device('cpu')):
    all_mask = list()
    for each_mg in mask_generator_list:
        temp_mask:Tensor = each_mg.get()
        if len(temp_mask.shape) == 2:
            temp_mask = temp_mask.unsqueeze(dim=-1)
        assert (
            len(temp_mask.shape) == 3
        ), "Unaccpetable static multihead mask"
        all_mask.append(temp_mask)
    all_mask = torch.cat(all_mask,dim=-1)
    assert (
        all_mask.shape[2] < num_head
    ), "Not enough multihead num"
    all_true_mask = torch.full(
        [all_mask.shape[0],all_mask.shape[1],num_head-all_mask.shape[2]],True)
    all_mask=torch.cat([all_mask,all_true_mask],dim=-1).contiguous().to(device)
    return all_mask

if __name__ == '__main__':
    

    # from torch_geometric_temporal.dataset import METRLADatasetLoader
    # loader = METRLADatasetLoader()
    # dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
    # for snapshot in dataset:
    #     edge_index = snapshot.edge_index
    #     edge_weight = snapshot.edge_weight
    #     num_nodes = snapshot.num_nodes
    #     break
    # dense_adj_mx = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight)
    # dense_adj_mx = dense_adj_mx.squeeze().numpy()
    # I = Infomap_Multi_Mask_Generator(num_node=num_nodes, graph_data=dense_adj_mx)
    
    # dense_adj_mx = np.array([[1.,0.,2.,-3.],[0.,1.,2.,-1.],[1.,1.,1.,-2.],[0.,1.,1,-1.]],dtype=np.float16)
    # temp1 = torch.from_numpy(dense_adj_mx)
    # temp2 = temp1.T
    # out = torch.where(temp1==temp2,
    #                     torch.tensor([True,]).expand([4,4]),
    #                     torch.tensor([False,]).expand([4,4]))
    # print(out)


    # dense_adj_mx = np.array([[1.,0.,2.,-3.],[0.,1.,2.,-1.],[1.,1.,1.,-2.],[0.,1.,torch.inf,-1.]],dtype=np.float16)
    # # out = dense_adj_mx>0 and dense_adj_mx is not torch.inf
    # dense_adj_mx = torch.from_numpy(dense_adj_mx)
    # out0 = dense_adj_mx>0
    # out1 = torch.where(dense_adj_mx==torch.inf,
    #                   torch.full_like(dense_adj_mx,False,dtype=torch.bool),
    #                   torch.full_like(dense_adj_mx,True,dtype=torch.bool))
    # print(out0 * out1)

    

    itr = [
        [0,1],
        [1,1],
        [1,2],
        [2,2]
    ]
    max_num_module = 2
    num_node = 3
    clu_tags = torch.full([max_num_module,num_node],-1,dtype=torch.int)
    for each in itr:
        # 一个行复制，一个列复制，用两个矩阵where相等为True,overlap部分无法处理(mend)
        clu_tags[each[1]-1][each[0]] = 1
    temp1 = clu_tags.unsqueeze(2).expand([max_num_module,num_node,num_node])
    temp2 = temp1.transpose(1,2)
    out = torch.any((temp1==temp2)*(temp1!=-1),dim=0)
    print(out)

    # dense_adj_mx = np.array([[1.,1.,1.,0.],[1.,1.,1.,0],[1.,1.,1.,0.],[0.,0.,0.,1.]],dtype=np.float16)
    # mg1 = Infomap_Multi_Mask_Generator(num_node=4,graph_data=dense_adj_mx)
    # print(mg1.get(),mg1.get().shape)
    # mg2 = Graph_Mask_Generator(num_node=4,graph_data=dense_adj_mx)
    # print(mg2.get(),mg2.get().shape)
    # mask = get_static_multihead_mask(num_head=8,mask_generator_list=[mg1,mg2])
    # print(mask.shape)