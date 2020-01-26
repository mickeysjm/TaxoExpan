import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax as dgl_edge_softmax
from dgl.nn.pytorch.glob import SumPooling, MaxPooling
import math

""" 
Graph Propagation Modules: GCN, GAT, PGCN, PGAT
"""
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout, bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)
        # normalization by square root of src degree
        h = h * g.ndata['norm']
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))
        h = g.ndata.pop('h')
        # normalization by square root of dst degree
        h = h * g.ndata['norm']
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=1, feat_drop=0.5, attn_drop=0.5, leaky_relu_alpha=0.2, residual=False):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x : x
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x : x
        self.attn_l = nn.Parameter(torch.Tensor(size=(1, num_heads, out_dim)))
        self.attn_r = nn.Parameter(torch.Tensor(size=(1, num_heads, out_dim)))
        nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_l.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_r.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_alpha)
        self.softmax = dgl_edge_softmax
        self.residual = residual
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
                nn.init.xavier_normal_(self.res_fc.weight.data, gain=1.414)
            else:
                self.res_fc = None

    def forward(self, g, feature):
        # prepare
        h = self.feat_drop(feature)  # NxD
        ft = self.fc(h).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
        a1 = (ft * self.attn_l).sum(dim=-1).unsqueeze(-1) # N x H x 1
        a2 = (ft * self.attn_r).sum(dim=-1).unsqueeze(-1) # N x H x 1
        g.ndata['ft'] = ft
        g.ndata['a1'] = a1
        g.ndata['a2'] = a2
        # 1. compute edge attention
        g.apply_edges(self.edge_attention)
        # 2. compute softmax
        self.edge_softmax(g)
        # 3. compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        g.update_all(fn.src_mul_edge('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))
        ret = g.ndata['ft']
        # 4. residual
        if self.residual:
            if self.res_fc is not None:
                resval = self.res_fc(h).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
            else:
                resval = torch.unsqueeze(h, 1)  # Nx1xD'
            ret = resval + ret
        return ret

    def edge_attention(self, edges):
        # an edge UDF to compute unnormalized attention values from src and dst
        a = self.leaky_relu(edges.src['a1'] + edges.dst['a2'])
        return {'a' : a}

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        # Dropout attention scores and save them
        g.edata['a_drop'] = self.attn_drop(attention)

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, activation, in_dropout=0.1, hidden_dropout=0.1, output_dropout=0.0):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(in_dim, hidden_dim, activation, in_dropout))
        # hidden layers
        for l in range(num_layers - 1):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim, activation, hidden_dropout))
        # output layer
        self.layers.append(GCNLayer(hidden_dim, out_dim, None, output_dropout))

    def forward(self, g, features):
        h = features
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(h.device)
        g.ndata['norm'] = norm.unsqueeze(1)
        for layer in self.layers:
            h = layer(g, h)
        return h

class PGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, pos_dim, num_layers, activation, in_dropout=0.1, hidden_dropout=0.1, output_dropout=0.0, position_vocab_size=3):
        super(PGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.prop_position_embeddings = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(in_dim+pos_dim, hidden_dim, activation, in_dropout))
        self.prop_position_embeddings.append(nn.Embedding(position_vocab_size, pos_dim))
        # hidden layers
        for l in range(num_layers - 1):
            self.layers.append(GCNLayer(hidden_dim+pos_dim, hidden_dim, activation, hidden_dropout))
            self.prop_position_embeddings.append(nn.Embedding(position_vocab_size, pos_dim))
        # output layer
        self.layers.append(GCNLayer(hidden_dim+pos_dim, out_dim, None, output_dropout))
        self.prop_position_embeddings.append(nn.Embedding(position_vocab_size, pos_dim))

    def forward(self, g, features):
        h = features
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(h.device)
        g.ndata['norm'] = norm.unsqueeze(1)

        positions = g.ndata.pop('pos').to(h.device)
        for idx, layer in enumerate(self.layers):
            p = self.prop_position_embeddings[idx](positions)
            h = layer(g, torch.cat((h, p), 1))
        return h

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, heads, activation, feat_drop=0.5, attn_drop=0.5, leaky_relu_alpha=0.2, residual=False):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input layer, no residual
        self.gat_layers.append(GATLayer(in_dim, hidden_dim, heads[0], feat_drop, attn_drop, leaky_relu_alpha, False))
        # hidden layers, due to multi-head, the in_dim = hidden_dim * num_heads
        for l in range(1, num_layers):
            self.gat_layers.append(GATLayer(hidden_dim * heads[l-1], hidden_dim, heads[l], feat_drop, attn_drop, leaky_relu_alpha, residual))
        # output layer
        self.gat_layers.append(GATLayer(hidden_dim * heads[-2], out_dim, heads[-1], feat_drop, attn_drop, leaky_relu_alpha, residual))

    def forward(self, g, features):
        h = features
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
            h = self.activation(h)
        # output projection
        h = self.gat_layers[-1](g, h).mean(1)
        return h

class PGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, pos_dim, num_layers, heads, activation, feat_drop=0.5, attn_drop=0.5, leaky_relu_alpha=0.2, residual=False, position_vocab_size=3):
        super(PGAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.prop_position_embeddings = nn.ModuleList()
        self.activation = activation
        # input layer, no residual
        self.gat_layers.append(GATLayer(in_dim+pos_dim, hidden_dim, heads[0], feat_drop, attn_drop, leaky_relu_alpha, False))
        self.prop_position_embeddings.append(nn.Embedding(position_vocab_size, pos_dim))
        # hidden layers, due to multi-head, the in_dim = hidden_dim * num_heads
        for l in range(1, num_layers):
            self.gat_layers.append(GATLayer(hidden_dim * heads[l-1] + pos_dim, hidden_dim, heads[l], feat_drop, attn_drop, leaky_relu_alpha, residual))
            self.prop_position_embeddings.append(nn.Embedding(position_vocab_size, pos_dim))
        # output layer
        self.gat_layers.append(GATLayer(hidden_dim * heads[-2] + pos_dim, out_dim, heads[-1], feat_drop, attn_drop, leaky_relu_alpha, residual))
        self.prop_position_embeddings.append(nn.Embedding(position_vocab_size, pos_dim))

    def forward(self, g, features):
        h = features
        positions = g.ndata.pop('pos').to(h.device)
        for l in range(self.num_layers):
            p = self.prop_position_embeddings[l](positions)
            h = self.gat_layers[l](g, torch.cat((h, p), 1)).flatten(1)
            h = self.activation(h)
        # output projection
        p = self.prop_position_embeddings[-1](positions)
        h = self.gat_layers[-1](g, torch.cat((h, p), 1)).mean(1)
        return h


""" 
Graph Readout Modules: MR, WMR, CR, [SumPooling, MaxPooling]
TODO: try GlobalAttentionPooling
"""
class MeanReadout(nn.Module):
    def __init__(self):
        super(MeanReadout, self).__init__()
    
    def forward(self, g, pos=None):
        return dgl.mean_nodes(g, 'h')
        
class WeightedMeanReadout(nn.Module):
    def __init__(self):
        super(WeightedMeanReadout, self).__init__()
        self.position_weights = nn.Embedding(3, 1)
        self.nonlinear = F.softplus
    
    def forward(self, g, pos):
        g.ndata['a'] = self.nonlinear(self.position_weights(pos))
        return dgl.mean_nodes(g, 'h', 'a')

class ConcatReadout(nn.Module):
    def __init__(self):
        super(ConcatReadout, self).__init__()
    
    def forward(self, g, pos):
        normalizer = torch.tensor(g.batch_num_nodes).unsqueeze_(1).float().to(pos.device)

        g.ndata['a_gp'] = (pos == 0).float()
        gp_embed = dgl.sum_nodes(g, 'h', 'a_gp') / normalizer
        g.ndata['a_p'] = (pos == 1).float()
        p_embed = dgl.mean_nodes(g, 'h', 'a_p')
        g.ndata['a_sib'] = (pos == 2).float()
        sib_embed = dgl.sum_nodes(g, 'h', 'a_sib') / normalizer
        
        return torch.cat((gp_embed, p_embed, sib_embed), 1)

class SumReadout(nn.Module):
    def __init__(self):
        super(SumReadout, self).__init__()
        self.sum_pooler = SumPooling()
    
    def forward(self, g):
        feat = g.ndata['h']
        return self.sum_pooler(g, feat)

class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()
        self.max_pooler = MaxPooling()
    
    def forward(self, g):
        feat = g.ndata['h']
        return self.max_pooler(g, feat)

""" 
Graph Matching Modules: MLP, LBM, [NTN]
"""
class MLP(nn.Module):
    def __init__(self, l_dim, r_dim, hidden_dim):
        super(MLP, self).__init__()
        activation = nn.ReLU()  
        self.ffn = nn.Sequential(
            nn.Linear(l_dim+r_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, e1, e2):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        return self.ffn(torch.cat((e1, e2), 1))


class BIM(nn.Module):
    def __init__(self, l_dim, r_dim):
        super(BIM, self).__init__()
        self.W = nn.Bilinear(l_dim, r_dim, 1, bias=False)
        
    def forward(self, e1, e2):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        return self.W(e1, e2)


class LBM(nn.Module):
    def __init__(self, l_dim, r_dim):
        super(LBM, self).__init__()
        self.W = nn.Bilinear(l_dim, r_dim, 1, bias=False)
        
    def forward(self, e1, e2):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        return torch.exp(self.W(e1, e2))


class NTN(nn.Module):
    def __init__(self, l_dim, r_dim, k=4, non_linear=F.tanh):
        super(NTN, self).__init__()
        self.u_R = nn.Linear(k, 1, bias=False)
        self.f = non_linear
        self.W = nn.Bilinear(l_dim, r_dim, k, bias=True)
        self.V = nn.Linear(l_dim+r_dim, k, bias=False)
        
    def forward(self, e1, e2):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        return self.u_R(self.f(self.W(e1, e2) + self.V(torch.cat((e1, e2), 1))))
