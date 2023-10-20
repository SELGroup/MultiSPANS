import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import to_dense_adj,dense_to_sparse,degree
import numpy as np
import scipy.sparse as sp
import math

# pos_encoding

def SinCosPosEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        print (f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


class Positional_Encoding(nn.Module):
    """
        general positional encoding layer
        return [len,d_model]
    """
    def __init__(self, pe_type, learn_pe, q_len, d_model, device=torch.device('cpu')):
        super(Positional_Encoding,self).__init__()
        # Positional encoding
        self.device = device
        self.pe_type = pe_type
        if pe_type == None:    # random pe , for measuring impact of pe
            W_pos = torch.empty((q_len, d_model))
            nn.init.uniform_(W_pos, -0.02, 0.02)
            learn_pe = False
        elif pe_type == 'zero': # 1 dim random pe
            W_pos = torch.empty((q_len, 1))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        elif pe_type == 'zeros': # n dim random pe
            W_pos = torch.empty((q_len, d_model))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        elif pe_type == 'normal' or pe_type == 'gauss':
            W_pos = torch.zeros((q_len, 1))
            torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
        elif pe_type == 'uniform':
            W_pos = torch.zeros((q_len, 1))
            nn.init.uniform_(W_pos, a=0.0, b=0.1)
        elif pe_type == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
        elif pe_type == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
        elif pe_type == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
        elif pe_type == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
        elif pe_type == 'sincos': W_pos = SinCosPosEncoding(q_len, d_model, normalize=True)
        elif self.__class__ is Positional_Encoding:
            raise ValueError(f"{pe_type} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
            'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
        else: W_pos = None
        if W_pos is not None:
            self.W_pos = nn.Parameter(W_pos, requires_grad=learn_pe).to(self.device)
    def forward(self):
        return self.W_pos


"""
    external encoding()
"""
class External_Encoding(nn.Module):
    '''
        External encoding
        output [batch, _, t_seq, embed_dim]
    '''
    def __init__(self, d_model,device):
        super().__init__()
        self.day_embedding = nn.Embedding(7 ,64)
        self.time_embedding = nn.Embedding(24*12, 64)

    def forward(self,x:Tensor):
        '''
        Args:
            x: [b, #node, #len, 11]
        Output:
            x: [b, #node, #len, 3]
            ext: [b, #node, #len, 64]
        '''
        day_info = torch.argmax(x[...,-7:], dim=-1)
        time_info = (x[...,-8:-7] * 288).int().squeeze(-1)
        x = x[...,:-8]
        # day_ebd = self.day_embedding(day_info)
        time_ebd = self.time_embedding(time_info)
        return x, time_ebd


class S_Positional_Encoding(Positional_Encoding):
    def __init__(self, pe_type, learn_pe, node_num, d_model, dim_red_rate=0.5, device=torch.device('cpu')):
        super(S_Positional_Encoding,self).__init__(pe_type, learn_pe, node_num, d_model, device)
        self.pe_type = pe_type
        if pe_type == 'laplacian':
            self.pe_encoder = LaplacianPE(round(node_num*dim_red_rate),d_model,device = self.device)
        elif pe_type == 'centrality':
            self.pe_encoder =  CentralityPE(node_num,d_model)
        else : raise ValueError(f"{pe_type} is not a valid spatial pe (positional encoder. Available types: 'laplacian','centrality','gauss'=='normal', \
            'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    
    def forward(self,adj_mx=None):
        if self.pe_type == 'laplacian': return self.pe_encoder(adj_mx)
        elif self.pe_type == 'centrality': return self.pe_encoder(adj_mx)
        else: return self.W_pos.to(self.device)



class LaplacianPE(nn.Module): # from [Dwivedi and Bresson, 2020] code from PDformer
    def __init__(self, lape_dim, embed_dim, learn_pe=False,device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.lape_dim = lape_dim
        self.learn_pe = learn_pe
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def _calculate_normalized_laplacian(self, adj):
        adj = sp.coo_matrix(adj)
        d = np.array(adj.sum(1))
        isolated_point_num = np.sum(np.where(d, 0, 1))
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        return normalized_laplacian, isolated_point_num

    def _calculate_random_walk_laplacian(self, adj):
        adj = sp.coo_matrix(adj)
        d = np.array(adj.sum(1))
        isolated_point_num = np.sum(np.where(d, 0, 1))
        d_inv = np.power(d, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        random_walk_mx = sp.eye(adj.shape[0]) - d_mat_inv.dot(adj).tocoo()
        return random_walk_mx, isolated_point_num

    def _cal_lape(self, dense_adj_mx):
        L, isolated_point_num = self._calculate_normalized_laplacian(dense_adj_mx)
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort()
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

        laplacian_pe:Tensor = torch.from_numpy(
            EigVec[:, isolated_point_num + 1: self.lape_dim + isolated_point_num + 1]
            ).float().to(self.device)
        laplacian_pe.require_grad = self.learn_pe
        return laplacian_pe

    def forward(self, adj_mx):
        lap_mx = self._cal_lape(adj_mx)
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx)
        return lap_pos_enc


class WLPE(nn.Module): #from graph-bert
    def __init__(self, n_dim, embed_dim):
        super().__init__()
        raise NotImplementedError
    def forward(self,x):
        raise NotImplementedError



class CentralityPE(nn.Module): # from Graphormer
    """
        for link (unweight) graph
    """
    def __init__(self, num_node, embed_dim, device=torch.device('cpu'),):
        super().__init__()
        self.device = device
        self.max_in_degree = num_node+1
        self.max_out_degree = num_node+1
        self.in_degree_encoder = nn.Embedding(self.max_in_degree, embed_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(self.max_out_degree, embed_dim, padding_idx=0)

    def forward(self, dense_adj_mx):
        (edge_index,_) = dense_to_sparse(torch.from_numpy(dense_adj_mx))
        outdegree = degree(edge_index[0]).to(self.device)
        indegree = degree(edge_index[1]).to(self.device)
        cen_pos_en = self.in_degree_encoder(indegree.long())+self.out_degree_encoder(outdegree.long())
        return cen_pos_en

if __name__ == '__main__':
    adj_mx = np.array([[1.,0.,2.,3.],[0.,1.,2.,1.],[1.,1.,1.,2.],[0.,1.,2.,1.]],dtype=np.float16)
    print(adj_mx)
    pe_encoder = CentralityPE(4,2)
    lap_pe = pe_encoder.forward(adj_mx)
    print(lap_pe)