import networkx as nx 
import dgl
import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader
import random
import pickle
import time
from tqdm import tqdm
from itertools import chain

BATCH_GRAPH_EDGE_LIMIT = 10000

class MAGDatasetSlim(object):
    def __init__(self, name, path):
        """ Raw dataset class for MAG dataset, slimed version, only allow for loading pickled dataset
        
        Parameters
        ----------
        name : str
            taxonomy name 
        path : str
            path to dataset, if raw=True, this is the directory path to dataset, if raw=False, this is the pickle path
        """
        self.name = name  # taxonomy name
        self.g_full = dgl.DGLGraph()  # full graph, including masked train/validation node indices
        self.vocab = []  # from node_id to human-readable concept string
        self.train_node_ids = []  # a list of train node_ids
        self.validation_node_ids = []  # a list of validation node_ids 
        self.test_node_ids = []  # a list of test node_ids
        self._load_dataset_pickled(path)

    def _load_dataset_pickled(self, pickle_path):
        with open(pickle_path, "rb") as fin:
            data = pickle.load(fin)
        
        self.name = data["name"]
        self.g_full = data["g_full"]
        self.vocab = data["vocab"]
        self.train_node_ids = data["train_node_ids"]
        self.validation_node_ids = data["validation_node_ids"]
        self.test_node_ids = data["test_node_ids"]

class EdgeDataset(Dataset):
    def __init__(self, graph_dataset, mode="train", negative_size=20):
        assert mode in ["train", "validation", "test"], "mode in EdgeDataset must be one of train, validation, and test"
        start = time.time()
        self.mode = mode
        self.negative_size = negative_size

        self.node_features = graph_dataset.g_full.ndata['x']
        self.vocab = graph_dataset.vocab
        self.full_graph = graph_dataset.g_full.to_networkx()
        
        if mode == "train":
            self.node_list = graph_dataset.train_node_ids
            self.graph = self.full_graph.subgraph(graph_dataset.train_node_ids).copy()
        elif mode == "validation":
            self.node_list = graph_dataset.validation_node_ids
            self.graph = self.full_graph.subgraph(graph_dataset.train_node_ids + graph_dataset.validation_node_ids).copy()
        else:
            self.node_list = graph_dataset.test_node_ids
            self.graph = self.full_graph.subgraph(graph_dataset.train_node_ids + graph_dataset.test_node_ids).copy()

        # remove root nodes
        roots = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]
        interested_node_set = set(self.node_list) - set(roots)
        self.node_list = list(interested_node_set)

        self.node2parents = {}
        self.node2masks = {}
        self.all_positions = set(graph_dataset.train_node_ids)  # each `position` is the candidate parent node of query element
        for node in tqdm(self.graph.nodes(), desc="generating intermediate data ..."):
            parents = [edge[0] for edge in self.graph.in_edges(node)]
            self.node2parents[node] = parents
            if node in interested_node_set:
                descendants = nx.descendants(self.graph, node)
                masks = set(list(descendants) + parents + [node] + roots)  
                self.node2masks[node] = masks

        # remove the edges between validation/test node ids with train graph
        edge_to_remove = []
        if mode == "validation":
            for node in graph_dataset.validation_node_ids:
                edge_to_remove.extend(list(self.graph.in_edges(node)))
            print(f"Remove {len(edge_to_remove)} edges between validation nodes and training nodes")
        elif mode == "test":
            for node in graph_dataset.test_node_ids:
                edge_to_remove.extend(list(self.graph.in_edges(node)))
            print(f"Remove {len(edge_to_remove)} edges between test nodes and training nodes")
        self.graph.remove_edges_from(edge_to_remove)

        # used for sampling negative poistions during train/validation stage
        self.pointer = 0
        self.queue = (graph_dataset.train_node_ids * 5).copy()

        end = time.time()
        print(f"Finish loading dataset ({end-start} seconds)")
        super(EdgeDataset, self).__init__()

    def __str__(self):
        return f"EdgeDataset mode:{self.mode}"

    def __len__(self):
        return len(self.node_list)

    def _get_negatives_from_queue(self, anchor_node, negative_size):
        if self.pointer == 0:
            random.shuffle(self.queue)
        
        while True:
            negatives = [ele for ele in self.queue[self.pointer: self.pointer+negative_size] if ele not in self.node2masks[anchor_node]]
            if len(negatives) > 0:
                break
        
        self.pointer += negative_size
        if self.pointer >= len(self.queue):
            self.pointer = 0
            
        return negatives

    def __getitem__(self, idx):
        """ 
        
        Parameters
        ----------
        idx : int
            sample index
        """
        res = []
        anchor_node = self.node_list[idx]

        # positive nodes
        res.extend([[positive, anchor_node, 1] for positive in self.node2parents[anchor_node]])
        
        # negative nodes
        if self.mode in ["train", "validation"]:
            negatives = self._get_negatives_from_queue(anchor_node, self.negative_size)
        else:
            negatives = [ele for ele in self.all_positions if ele not in self.node2masks[anchor_node]]
        res.extend([[negative, anchor_node, 0] for negative in negatives])

        return tuple(res)

class SubGraphDataset(EdgeDataset):
    def __init__(self, graph_dataset, mode="train", negative_size=20, UNK_idx=None):
        super(SubGraphDataset, self).__init__(graph_dataset, mode, negative_size)
        self.UNK_idx = len(self.vocab)
    
    def __str__(self):
        return f"SubGraphDataset mode:{self.mode}"

    def __getitem__(self, idx):
        res = []
        anchor_node = self.node_list[idx]

        # positive examples
        for parent in self.node2parents[anchor_node]:
            siblings = [edge[1] for edge in self.graph.out_edges(parent) if edge[1] != anchor_node]
            if len(siblings) <= 50:
                res.append([parent, siblings+[self.UNK_idx]*(50-len(siblings)), anchor_node, 1])
            else:
                random.shuffle(siblings)
                res.append([parent, siblings[:50], anchor_node, 1])
        
        # negative examples
        if self.mode in ["train", "validation"]:
            negatives = self._get_negatives_from_queue(anchor_node, self.negative_size)
        else:
            negatives = [ele for ele in self.all_positions if ele not in self.node2masks[anchor_node]]
        for negative in negatives:
            siblings = [edge[1] for edge in self.graph.out_edges(negative)]
            if len(siblings) <= 50:
                res.append([negative, siblings+[self.UNK_idx]*(50-len(siblings)), anchor_node, 0])
            else:
                random.shuffle(siblings)
                res.append([negative, siblings[:50], anchor_node, 0])

        return tuple(res)

class AnchorParentDataset(EdgeDataset):
    def __init__(self, graph_dataset, mode="train", negative_size=20, UNK_idx=None):
        super(AnchorParentDataset, self).__init__(graph_dataset, mode, negative_size)
        self.UNK_idx = len(self.vocab)
    
    def __str__(self):
        return f"AnchorParentDataset mode:{self.mode}"
   
    def __getitem__(self, idx):
        res = []
        anchor_node = self.node_list[idx]

        # positive examples
        for parent in self.node2parents[anchor_node]:
            siblings = [edge[1] for edge in self.graph.out_edges(parent) if edge[1] != anchor_node]
            if len(siblings) <= 100:
                siblings += [self.UNK_idx]*(100-len(siblings))
            else:
                random.shuffle(siblings)
                siblings = siblings[:100]
            
            grand_parents = [edge[0] for edge in self.graph.in_edges(parent)]
            if len(grand_parents) <= 100:
                grand_parents += [self.UNK_idx]*(100-len(grand_parents))
            else:
                random.shuffle(grand_parents)
                grand_parents = grand_parents[:100]

            res.append([parent, siblings, grand_parents, anchor_node, 1])

        # negative examples
        if self.mode in ["train", "validation"]:
            num_negatives = min(self.negative_size * len(res), len(self.node2negative_pool[anchor_node]))
            negatives = random.choices(self.node2negative_pool[anchor_node], k=num_negatives)
        else:
            negatives = self.node2negative_pool[anchor_node]
        for negative in negatives:
            siblings = [edge[1] for edge in self.graph.out_edges(negative)]
            if len(siblings) <= 100:
                siblings += [self.UNK_idx]*(100-len(siblings))
            else:
                random.shuffle(siblings)
                siblings = siblings[:100]
            
            grand_parents = [edge[0] for edge in self.graph.in_edges(negative)]
            if len(grand_parents) <= 100:
                grand_parents += [self.UNK_idx]*(100-len(grand_parents))
            else:
                random.shuffle(grand_parents)
                grand_parents = grand_parents[:100]

            res.append([negative, siblings, grand_parents, anchor_node, 0])

        return tuple(res)

def collate_edge_small_batch(samples):
    parents, children, labels = map(list, zip(*chain(*samples)))
    return torch.tensor(parents), torch.tensor(children), torch.tensor(labels)

def collate_edge_large_batch(samples):
    parents, children, labels = map(list, zip(*chain(*samples)))
    batched_parents = []
    batched_children = []
    bached_labels = []
    start = 0
    while start < len(parents):
        batched_parents.append(torch.tensor(parents[start: start+BATCH_GRAPH_EDGE_LIMIT]))
        batched_children.append(torch.tensor(children[start: start+BATCH_GRAPH_EDGE_LIMIT]))
        bached_labels.append(torch.tensor(labels[start: start+BATCH_GRAPH_EDGE_LIMIT]))
        start += BATCH_GRAPH_EDGE_LIMIT
    return batched_parents, batched_children, bached_labels

def collate_subgraph_small_batch(samples):
    parents, siblings, children, labels = map(list, zip(*chain(*samples)))
    return torch.tensor(parents), torch.tensor(siblings), torch.tensor(children), torch.tensor(labels)

def collate_subgraph_large_batch(samples):
    parents, siblings, children, labels = map(list, zip(*chain(*samples)))
    batched_parents = []
    batched_siblings = []
    batched_children = []
    bached_labels = []
    start = 0
    while start < len(parents):
        batched_parents.append(torch.tensor(parents[start: start+BATCH_GRAPH_EDGE_LIMIT]))
        batched_siblings.append(torch.tensor(siblings[start: start+BATCH_GRAPH_EDGE_LIMIT]))
        batched_children.append(torch.tensor(children[start: start+BATCH_GRAPH_EDGE_LIMIT]))
        bached_labels.append(torch.tensor(labels[start: start+BATCH_GRAPH_EDGE_LIMIT]))
        start += BATCH_GRAPH_EDGE_LIMIT
    return batched_parents, batched_siblings, batched_children, bached_labels

def collate_apgraph_small_batch(samples):
    parents, siblings, grand_parents, children, labels = map(list, zip(*chain(*samples)))
    return torch.tensor(parents), torch.tensor(siblings), torch.tensor(grand_parents), torch.tensor(children), torch.tensor(labels)

def collate_apgraph_large_batch(samples):
    parents, siblings, grand_parents, children, labels = map(list, zip(*chain(*samples)))
    batched_parents = []
    batched_siblings = []
    batched_grand_parents = []
    batched_children = []
    bached_labels = []
    start = 0
    while start < len(parents):
        batched_parents.append(torch.tensor(parents[start: start+BATCH_GRAPH_EDGE_LIMIT]))
        batched_siblings.append(torch.tensor(siblings[start: start+BATCH_GRAPH_EDGE_LIMIT]))
        batched_grand_parents.append(torch.tensor(grand_parents[start: start+BATCH_GRAPH_EDGE_LIMIT]))
        batched_children.append(torch.tensor(children[start: start+BATCH_GRAPH_EDGE_LIMIT]))
        bached_labels.append(torch.tensor(labels[start: start+BATCH_GRAPH_EDGE_LIMIT]))
        start += BATCH_GRAPH_EDGE_LIMIT
    return batched_parents, batched_siblings, batched_grand_parents, batched_children, bached_labels

class EdgeDataLoader(DataLoader):
    def __init__(self, data_path, mode, batch_type="small_batch", batch_size=10, negative_size=20, shuffle=True, num_workers=4):
        assert batch_type in ["small_batch", "large_batch"], "batch_type arg must be either small_batch or large_batch"
        assert mode in ["train", "validation", "test"], "mode must be one of train, validation, and test"
        raw_graph_dataset = MAGDatasetSlim(name="", path=data_path)
        edge_dataset = EdgeDataset(raw_graph_dataset, mode=mode, negative_size=negative_size)
        self.dataset = edge_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        if batch_type == "small_batch":
            self.collate_fn = collate_edge_small_batch
        else:
            self.collate_fn = collate_edge_large_batch
        self.num_workers = num_workers
        super(EdgeDataLoader, self).__init__(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn, num_workers=self.num_workers)
        self.n_samples = len(self.dataset)  # total number of samples that will be emitted by this data loader
        self.mode = self.dataset.mode

class SubGraphDataLoader(DataLoader):
    def __init__(self, data_path, mode, batch_type="small_batch", batch_size=10, negative_size=20, shuffle=True, num_workers=8):
        assert batch_type in ["small_batch", "large_batch"], "batch_type arg must be either small_batch or large_batch"
        assert mode in ["train", "validation", "test"], "mode must be one of train, validation, and test"
        raw_graph_dataset = MAGDatasetSlim(name="", path=data_path)
        subgraph_dataset = SubGraphDataset(raw_graph_dataset, mode=mode, negative_size=negative_size)
        self.dataset = subgraph_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        if batch_type == "small_batch":
            self.collate_fn = collate_subgraph_small_batch
        else:
            self.collate_fn = collate_subgraph_large_batch
        self.num_workers = num_workers
        super(SubGraphDataLoader, self).__init__(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn, num_workers=self.num_workers)
        self.n_samples = len(self.dataset)  # total number of samples that will be emitted by this data loader
        self.mode = self.dataset.mode

class AnchorParentDataLoader(DataLoader):
    def __init__(self, data_path, mode, batch_type="small_batch", batch_size=10, negative_size=20, shuffle=True, num_workers=8):
        assert batch_type in ["small_batch", "large_batch"], "batch_type arg must be either small_batch or large_batch"
        assert mode in ["train", "validation", "test"], "mode must be one of train, validation, and test"
        raw_graph_dataset = MAGDatasetSlim(name="", path=data_path)
        apgraph_dataset = AnchorParentDataset(raw_graph_dataset, mode=mode, negative_size=negative_size)
        self.dataset = apgraph_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        if batch_type == "small_batch":
            self.collate_fn = collate_apgraph_small_batch
        else:
            self.collate_fn = collate_apgraph_large_batch
        self.num_workers = num_workers
        super(AnchorParentDataLoader, self).__init__(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn, num_workers=self.num_workers)
        self.n_samples = len(self.dataset)  # total number of samples that will be emitted by this data loader
        self.mode = self.dataset.mode
