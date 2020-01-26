from torch.utils.data import DataLoader
from .dataset import MAGDataset, MaskedGraphDataset
import dgl
import torch
from itertools import chain 

BATCH_GRAPH_NODE_LIMIT = 100000

def collate_graph_and_node_small_batch(samples):
    """ Batch a small list of (graph, node_feature, label) triplets, used in training/validation stage
    
    Parameters
    ----------
    samples : list 
        a list of lists of (dgl.DGLGraph, node_feature, label) triplets
    
    Returns
    -------
    batched_graph : dgl.BatchDGLGraph
        a single batched dgl.BatchDGLGraph object
    batched_label : tensor
        a label tensor corresponds to the batch of graphs 
    """
    graphs, node_features, labels = map(list, zip(*chain(*samples)))
    batched_graph = dgl.batch(graphs)
    batched_node_features = torch.stack(node_features)
    batched_label = torch.tensor(labels)
    return batched_graph, batched_node_features, batched_label


def collate_graph_and_node_large_batch(samples):
    """ Batch a large list of (graph, node_feature, label) triplets, used in test stage
    
    Parameters
    ----------
    samples : list
        a list of lists of (dgl.DGLGraph, node_feature, label) triplets
    
    Returns
    -------
    batched_graphs : list
        a list of batched dgl.BatchDGLGraph objects
    batched_labels : list
        a list of label tensors
    """
    graphs, node_features, labels = map(list, zip(*chain(*samples)))
    batched_graphs = []
    batched_node_features = []
    batched_labels = []
    number_of_nodes_in_batch = 0
    gs = []  # current graphs
    fs = []  # current node features
    ls = []  # current labels
    for i, graph in enumerate(graphs):
        gs.append(graph)
        fs.append(node_features[i])
        ls.append(labels[i])
        number_of_nodes_in_batch += graph.number_of_nodes()
        if number_of_nodes_in_batch > BATCH_GRAPH_NODE_LIMIT and (len(gs) > 1):
            batched_graphs.append(dgl.batch(gs))
            batched_node_features.append(torch.stack(fs))
            batched_labels.append(torch.tensor(ls))
            number_of_nodes_in_batch = 0
            gs = []
            fs = []
            ls = []
    if len(gs) != 0:  # add the last batch
        batched_graphs.append(dgl.batch(gs))
        batched_node_features.append(torch.stack(fs))
        batched_labels.append(torch.tensor(ls))

    return batched_graphs, batched_node_features, batched_labels


class MaskedGraphDataLoader(DataLoader):
    def __init__(self, mode, data_path, sampling_mode=1, batch_size=10, batch_type="small_batch", negative_size=20, expand_factor=50, shuffle=True, num_workers=8, cache_refresh_time=64, normalize_embed=False, test_topk=-1):
        assert batch_type in ["small_batch", "large_batch"], "batch_type arg must be either small_batch or large_batch"
        assert mode in ["train", "validation", "test"], "mode must be one of train, validation, and test"

        self.mode = mode
        self.sampling_mode = sampling_mode
        self.batch_size = batch_size
        self.batch_type = batch_type
        self.negative_size = negative_size
        self.expand_factor = expand_factor
        self.shuffle = shuffle
        self.cache_refresh_time = cache_refresh_time
        self.normalize_embed = normalize_embed

        raw_graph_dataset = MAGDataset(name="", path=data_path, raw=False)
        msk_graph_dataset = MaskedGraphDataset(raw_graph_dataset, mode=mode, sampling_mode=sampling_mode, negative_size=negative_size, expand_factor=expand_factor, cache_refresh_time=cache_refresh_time, normalize_embed=normalize_embed, test_topk=test_topk)
        self.dataset = msk_graph_dataset
        if self.batch_type == "small_batch":
            self.collate_fn = collate_graph_and_node_small_batch
        else:
            self.collate_fn = collate_graph_and_node_large_batch
        self.num_workers = num_workers
        super(MaskedGraphDataLoader, self).__init__(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn, num_workers=self.num_workers, pin_memory=True)
        self.n_samples = len(self.dataset)  # total number of samples that will be emitted by this data loader
        
    def __str__(self):
        return "\n\t".join([
            f"MaskedGraphDataLoader mode: {self.mode}",
            f"sampling_mode: {self.sampling_mode}",
            f"batch_size: {self.batch_size}",
            f"negative_size: {self.negative_size}",
            f"expand_factor: {self.expand_factor}",
            f"cache_refresh_time: {self.cache_refresh_time}",
            f"normalize_embed: {self.normalize_embed}",
        ])
