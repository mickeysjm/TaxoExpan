import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from base import BaseModel
import ipdb
import math
from .model_zoo import GCN, GAT, PGCN, PGAT, MeanReadout, WeightedMeanReadout, ConcatReadout, MLP, BIM, LBM


class TaxoExpan(BaseModel):
    """ return getattr(module, module_cfg['type'])(*args, **module_cfg['args'])
    """
    def __init__(self, propagation_method, readout_method, matching_method, **options):
        super(TaxoExpan, self).__init__()
        self.propagation_method = propagation_method
        self.readout_method = readout_method
        self.matching_method = matching_method
        self.options = options
        if propagation_method == "GCN":
            self.graph_propagate = GCN(
                options["in_dim"], options["hidden_dim"], options["out_dim"], num_layers=options["num_layers"], 
                activation=F.leaky_relu, in_dropout=options["feat_drop"], hidden_dropout=options["hidden_drop"], 
                output_dropout=options["out_drop"])
        elif propagation_method == "PGCN":
            self.graph_propagate = PGCN(
                options["in_dim"], options["hidden_dim"], options["out_dim"], options["pos_dim"], 
                num_layers=options["num_layers"], activation=F.leaky_relu, in_dropout=options["feat_drop"], 
                hidden_dropout=options["hidden_drop"], output_dropout=options["out_drop"])
        elif propagation_method == "GAT":
            self.graph_propagate = GAT(
                options["in_dim"], options["hidden_dim"], options["out_dim"], num_layers=options["num_layers"], 
                heads=options["heads"], activation=F.leaky_relu, feat_drop=options["feat_drop"], 
                attn_drop=options["attn_drop"])
        elif propagation_method == "PGAT":
             self.graph_propagate = PGAT(
                options["in_dim"], options["hidden_dim"], options["out_dim"], options["pos_dim"], 
                num_layers=options["num_layers"], heads=options["heads"], activation=F.leaky_relu, 
                feat_drop=options["feat_drop"], attn_drop=options["attn_drop"])
        else:
            assert f"Unacceptable Graph Propagation Method: {propagation_method}"

        if readout_method == "MR":
            self.readout = MeanReadout()
            l_dim = options["out_dim"]
            r_dim = options["in_dim"]
        elif readout_method == "WMR":
            self.readout = WeightedMeanReadout()
            l_dim = options["out_dim"]
            r_dim = options["in_dim"]
        elif readout_method == "CR":
            self.readout = ConcatReadout()
            l_dim = options["out_dim"]*3
            r_dim = options["in_dim"]
        else:
            assert f"Unacceptable Readout Method: {readout_method}"

        if matching_method == "MLP":
            self.match = MLP(l_dim, r_dim, options["hidden_dim"])
        elif matching_method == "LBM":
            self.match = LBM(l_dim, r_dim)
        elif matching_method == "BIM":
            self.match = BIM(l_dim, r_dim)
        else:
            assert f"Unacceptable Matching Method: {matching_method}"


    def forward(self, g, h, qf):
        """ TaxoExpan forward pass
        
        Parameters
        ----------
        g : dgl.BatchedGraph
        h : node feature in batched g
        qf : query feature
        
        Returns
        -------
        scores: matching score of each node in g and query 
        """
        pos = g.ndata['pos'].to(h.device)
        g.ndata['h'] = self.graph_propagate(g, h)
        hg = self.readout(g, pos)
        scores = self.match(hg, qf)
        return scores     
