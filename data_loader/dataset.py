import networkx as nx 
import dgl
from gensim.models import KeyedVectors
import numpy as np 
import torch 
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
import pickle
import time
from tqdm import tqdm
import random
import copy
from itertools import chain
import os


class Taxon(object):
    def __init__(self, tx_id, rank=-1, norm_name="none", display_name="None", main_type="", level="-100", p_count=0, c_count=0, create_date="None"):
        self.tx_id = tx_id
        self.rank = int(rank)
        self.norm_name = norm_name
        self.display_name = display_name
        self.main_type = main_type
        self.level = int(level)
        self.p_count = int(p_count)
        self.c_count = int(c_count)
        self.create_date = create_date
        
    def __str__(self):
        return "Taxon {} (name: {}, level: {})".format(self.tx_id, self.norm_name, self.level)
        
    def __lt__(self, another_taxon):
        if self.level < another_taxon.level:
            return True
        else:
            return self.rank < another_taxon.rank


class MAGDataset(object):
    def __init__(self, name, path, embed_suffix="", raw=True, existing_partition=False):
        """ Raw dataset class for MAG dataset
        
        Parameters
        ----------
        name : str
            taxonomy name 
        path : str
            path to dataset, if raw=True, this is the directory path to dataset, if raw=False, this is the pickle path
        embed_suffix : str
            suffix of embedding file name, by default ""
        raw : bool, optional
            load raw dataset from txt (True) files or load pickled dataset (False), by default True
        existing_partition : bool, optional
            whether to use the existing the train/validation/test partitions or randomly sample new ones, by default False
        """
        self.name = name  # taxonomy name
        self.embed_suffix = embed_suffix
        self.existing_partition = existing_partition
        self.g_full = dgl.DGLGraph()  # full graph, including masked train/validation node indices
        self.vocab = []  # from node_id to human-readable concept string
        self.train_node_ids = []  # a list of train node_ids
        self.validation_node_ids = []  # a list of validation node_ids 
        self.test_node_ids = []  # a list of test node_ids
        
        if raw:
            self._load_dataset_raw(path)
        else:
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

    def _load_dataset_raw(self, dir_path):
        """ Load data from three seperated files, generate train/validation/test partitions, and save to binary pickled dataset.
        Please refer to the README.md file for details.


        Parameters
        ----------
        dir_path : str
            The path to a directory containing three input files. 
        """
        node_file_name = os.path.join(dir_path, f"{self.name}.terms")
        edge_file_name = os.path.join(dir_path, f"{self.name}.taxo")
        if self.embed_suffix == "":
            embedding_file_name = os.path.join(dir_path, f"{self.name}.terms.embed")
            output_pickle_file_name = os.path.join(dir_path, f"{self.name}.pickle.bin")
        else:
            embedding_file_name = os.path.join(dir_path, f"{self.name}.terms.{self.embed_suffix}.embed")
            output_pickle_file_name = os.path.join(dir_path, f"{self.name}.{self.embed_suffix}.pickle.bin")
        if self.existing_partition:
            train_node_file_name = os.path.join(dir_path, f"{self.name}.terms.train")
            validation_node_file_name = os.path.join(dir_path, f"{self.name}.terms.validation")
            test_file_name = os.path.join(dir_path, f"{self.name}.terms.test")
        
        tx_id2taxon = {}
        taxonomy = nx.DiGraph()

        # load nodes
        with open(node_file_name, "r") as fin:
            for line in tqdm(fin, desc="Loading terms"):
                line = line.strip()
                if line:
                    segs = line.split("\t")
                    assert len(segs) == 2, f"Wrong number of segmentations {line}"
                    taxon = Taxon(tx_id=segs[0], norm_name=segs[1], display_name=segs[1])
                    tx_id2taxon[segs[0]] = taxon
                    taxonomy.add_node(taxon)

        # load edges
        with open(edge_file_name, "r") as fin:
            for line in tqdm(fin, desc="Loading relations"):
                line = line.strip()
                if line:
                    segs = line.split("\t")
                    assert len(segs) == 2, f"Wrong number of segmentations {line}"
                    parent_taxon = tx_id2taxon[segs[0]]
                    child_taxon = tx_id2taxon[segs[1]]
                    taxonomy.add_edge(parent_taxon, child_taxon)

        # load embedding features
        print("Loading embedding ...")
        embeddings = KeyedVectors.load_word2vec_format(embedding_file_name)
        print(f"Finish loading embedding of size {embeddings.vectors.shape}")

        # load train/validation/test partition files if needed
        if self.existing_partition:
            print("Loading existing train/validation/test partitions")
            raw_train_node_list = self._load_node_list(train_node_file_name)
            raw_validation_node_list = self._load_node_list(validation_node_file_name)
            raw_test_node_list = self._load_node_list(test_file_name)
            
        # generate vocab, tx_id is the old taxon_id read from {self.name}.terms file, node_id is the new taxon_id from 0 to len(vocab)
        tx_id2node_id = {node.tx_id:idx for idx, node in enumerate(taxonomy.nodes()) } 
        node_id2tx_id = {v:k for k, v in tx_id2node_id.items()}
        self.vocab = [tx_id2taxon[node_id2tx_id[node_id]].norm_name + "@@@" + str(node_id) for node_id in node_id2tx_id]

        # generate dgl.DGLGraph() 
        edges = []
        for edge in taxonomy.edges():
            parent_node_id = tx_id2node_id[edge[0].tx_id]
            child_node_id = tx_id2node_id[edge[1].tx_id]
            edges.append([parent_node_id, child_node_id])

        node_features = np.zeros(embeddings.vectors.shape)
        for node_id, tx_id in node_id2tx_id.items():
            node_features[node_id, :] = embeddings[tx_id]
        node_features = torch.FloatTensor(node_features)

        self.g_full.add_nodes(len(node_id2tx_id), {'x': node_features})
        self.g_full.add_edges([e[0] for e in edges], [e[1] for e in edges])

        # generate validation/test node_indices using either existing partitions or randomly sampled partition
        if self.existing_partition:
            self.train_node_ids = [tx_id2node_id[tx_id] for tx_id in raw_train_node_list]
            self.validation_node_ids = [tx_id2node_id[tx_id] for tx_id in raw_validation_node_list]
            self.test_node_ids = [tx_id2node_id[tx_id] for tx_id in raw_test_node_list]
        else:
            leaf_node_ids = []
            for node in taxonomy.nodes():
                if taxonomy.out_degree(node) == 0:
                    leaf_node_ids.append(tx_id2node_id[node.tx_id])
            
            random.seed(47)
            random.shuffle(leaf_node_ids)
            validation_size = int(len(leaf_node_ids) * 0.1)
            test_size = int(len(leaf_node_ids) * 0.1)
            self.validation_node_ids = leaf_node_ids[:validation_size]
            self.test_node_ids = leaf_node_ids[validation_size:(validation_size+test_size)]
            self.train_node_ids = [node_id for node_id in node_id2tx_id if node_id not in self.validation_node_ids and node_id not in self.test_node_ids]

        # save to pickle for faster loading next time
        print("start saving pickle data")
        with open(output_pickle_file_name, 'wb') as fout:
            # Pickle the 'data' dictionary using the highest protocol available.
            data = {
                "name": self.name,
                "g_full": self.g_full,
                "vocab": self.vocab,
                "train_node_ids": self.train_node_ids,
                "validation_node_ids": self.validation_node_ids,
                "test_node_ids": self.test_node_ids,
            }
            pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)
        print(f"Save pickled dataset to {output_pickle_file_name}")

    def _load_node_list(self, file_path):
        node_list = []
        with open(file_path, "r") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    node_list.append(line)
        return node_list


class MaskedGraphDataset(Dataset):
    def __init__(self, graph_dataset, mode="train", sampling_mode=1, negative_size=32, expand_factor=64, cache_refresh_time=128, normalize_embed=False, test_topk=-1):
        assert mode in ["train", "validation", "test"], "mode in MaskedGraphDataset must be one of train, validation, and test"
        assert sampling_mode in [0,1,2,3], "sampling_mode in MaskedGraphDataset must be in [0,1,2,3]"
        if mode == "test":
            assert sampling_mode == 0, "!!! During testing, sampling_mode must be 0, in order to emit all positive true parents"    
        start = time.time()
        self.mode = mode
        self.sampling_mode = sampling_mode
        self.negative_size = negative_size
        self.expand_factor = expand_factor
        self.cache_refresh_time = cache_refresh_time
        self.normalize_embed = normalize_embed
        self.test_topk = test_topk

        self.node_features = graph_dataset.g_full.ndata['x']
        if self.normalize_embed:
            self.node_features = F.normalize(self.node_features, p=2, dim=1)
        self.vocab = graph_dataset.vocab
        self.full_graph = graph_dataset.g_full.to_networkx()
        
        # add node feature vector
        self.kv = KeyedVectors(vector_size=self.node_features.shape[1])
        self.kv.add([str(i) for i in range(len(self.vocab))], self.node_features.numpy())

        # add interested node list and subgraph
        if mode == "train":
            self.node_list = graph_dataset.train_node_ids
            self.graph = self.full_graph.subgraph(graph_dataset.train_node_ids).copy()
        elif mode == "validation":
            self.node_list = graph_dataset.validation_node_ids
            self.graph = self.full_graph.subgraph(graph_dataset.train_node_ids + graph_dataset.validation_node_ids).copy()
        else:
            self.node_list = graph_dataset.test_node_ids
            self.graph = self.full_graph.subgraph(graph_dataset.train_node_ids + graph_dataset.test_node_ids).copy()

        # remove supersource nodes (i.e., nodes without in-degree 0)
        roots = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]
        interested_node_set = set(self.node_list) - set(roots)
        self.node_list = list(interested_node_set)

        # generate and cache intermediate data
        self.node2parents = {}  # list of correct parent positions
        self.node2positive_pointer = {}  
        self.node2masks = {}  # self.node2masks[n] is a list of positions that should not be chosen as the anchor of query node n
        self.all_positions = set(graph_dataset.train_node_ids)  # each `position` is the candidate parent node of query element
        for node in tqdm(self.graph.nodes(), desc="generating intermediate data ..."):
            parents = [edge[0] for edge in self.graph.in_edges(node)]
            self.node2parents[node] = parents
            self.node2positive_pointer[node] = 0
            if node in interested_node_set:
                descendants = nx.descendants(self.graph, node)
                masks = set(list(descendants) + parents + [node] + roots)  
                self.node2masks[node] = masks

        # [IMPORTANT] Here, we must remove the edges between validation/test node ids with train graph to avoid data leakage
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

        # used for caching local subgraphs
        self.cache = {}  # if g = self.cache[anchor_node], then g is the egonet centered on the anchor_node
        self.cache_counter = {}  # if n = self.cache[anchor_node], then n is the number of times you used this cache

        # used for sampling negative poistions during train/validation stage
        self.pointer = 0
        self.queue = (graph_dataset.train_node_ids * 5).copy()

        end = time.time()
        print(f"Finish loading dataset ({end-start} seconds)")

    def __str__(self):
        return f"MaskedGraphDataset mode:{self.mode}"

    def __len__(self):
        return len(self.node_list)

    def __getitem__(self, idx):
        """ Generate an data instance based on train/validation/test mode.
        
        One data instance is a list of (anchor_egonet, query_node_feature, label) triplets.

        If self.sampling_mode == 0:
            This list may contain more than one triplets with label = 1
        If self.sampling_mode == 1:
            This list contain one and ONLY one triplet with label = 1, others have label = 0
        """
        res = []
        query_node = self.node_list[idx]
        
        # generate positive triplet(s)
        if self.sampling_mode == 0:
            for parent_node in self.node2parents[query_node]:
                anchor_egonet, query_node_feature = self._get_subgraph_and_node_pair(query_node, parent_node, 1)
                res.append([anchor_egonet, query_node_feature, 1])
        elif self.sampling_mode == 1:
            positive_pointer = self.node2positive_pointer[query_node]
            parent_node = self.node2parents[query_node][positive_pointer]
            anchor_egonet, query_node_feature = self._get_subgraph_and_node_pair(query_node, parent_node, 1)
            self.node2positive_pointer[query_node] = (positive_pointer+1) % len(self.node2parents[query_node])
            res.append([anchor_egonet, query_node_feature, 1])

        # select negative parents
        if self.mode in ["train", "validation"]:
            negative_parents = self._get_negative_anchors(query_node, self.negative_size)
        else:
            if self.test_topk == -1:
                negative_parents = [ele for ele in self.all_positions if ele not in self.node2masks[query_node]]
            else:
                negative_pool = [str(ele) for ele in self.all_positions if ele not in self.node2masks[query_node]]
                negative_dist = self.kv.distances(str(query_node), negative_pool)
                top_negatives = sorted(zip(negative_pool, negative_dist), key=lambda x:x[1])[:self.test_topk]
                negative_parents = [int(ele[0]) for ele in top_negatives]
        
        # generate negative triplets
        for negative_parent in negative_parents:
            anchor_egonet, query_node_feature = self._get_subgraph_and_node_pair(query_node, negative_parent, 0)
            res.append([anchor_egonet, query_node_feature, 0])

        return tuple(res)
    
    def _get_negative_anchors(self, query_node, negative_size):
        if self.sampling_mode == 0:
            return self._get_at_most_k_negatives(query_node, negative_size)
        elif self.sampling_mode == 1:
            return self._get_exactly_k_negatives(query_node, negative_size)

    def _get_at_most_k_negatives(self, query_node, negative_size):
        """ Generate AT MOST negative_size samples for the query node
        """
        if self.pointer == 0:
            random.shuffle(self.queue)
        
        while True:
            negatives = [ele for ele in self.queue[self.pointer: self.pointer+negative_size] if ele not in self.node2masks[query_node]]
            if len(negatives) > 0:
                break
        
        self.pointer += negative_size
        if self.pointer >= len(self.queue):
            self.pointer = 0
            
        return negatives

    def _get_exactly_k_negatives(self, query_node, negative_size):
        """ Generate EXACTLY negative_size samples for the query node
        """
        if self.pointer == 0: 
            random.shuffle(self.queue)
        
        masks = self.node2masks[query_node]
        negatives = []
        max_try = 0
        while len(negatives) != negative_size:
            n_lack = negative_size - len(negatives)
            negatives.extend([ele for ele in self.queue[self.pointer: self.pointer+n_lack] if ele not in masks])
            self.pointer += n_lack
            if self.pointer >= len(self.queue):
                self.pointer = 0
                random.shuffle(self.queue)
            max_try += 1
            if max_try > 10:  # corner cases, trim/expand negatives to the size
                print(f"Alert in _get_exactly_k_negatives, query_node: {query_node}, current negative size: {len(negatives)}")
                if len(negatives) > negative_size:
                    negatives = negatives[:negative_size]
                else:
                    negatives.extend([ele for ele in self.queue[: (negative_size-len(negatives))]])

        return negatives

    def _get_subgraph_and_node_pair(self, query_node, anchor_node, instance_mode):
        """ Generate anchor_egonet and obtain query_node feature

        instance_mode: 0 means negative example, 1 means positive example
        """
        # query_node_feature 
        query_node_feature = self.node_features[query_node, :]

        # [IMPORTANT] only read from cache if this pair is a negative example and already saved in cache
        # You cannot read the positive egonet from the cache because this egonet will contain the query node itself, which makes the model prediction task trivial
        if instance_mode == 0 and (anchor_node in self.cache) and (self.cache_counter[anchor_node] < self.cache_refresh_time):
            g = self.cache[anchor_node]
            self.cache_counter[anchor_node] += 1
        else:
            g = self._get_subgraph(query_node, anchor_node, instance_mode)
            if instance_mode == 0:  # save to cache
                self.cache[anchor_node] = g
                self.cache_counter[anchor_node] = 0
    
        return g, query_node_feature

    def _get_subgraph(self, query_node, anchor_node, instance_mode):
        # grand parents of query node (i.e., parents of anchor node)
        nodes = [edge[0] for edge in self.graph.in_edges(anchor_node)]  
        nodes_pos = [0] * len(nodes)
        
        # parent of query  (i.e., anchor node itself)
        parent_node_idx = len(nodes)
        nodes.append(anchor_node)  
        nodes_pos.append(1)
        
        # siblings of query node (i.e., children of anchor node)
        if instance_mode == 0:  # negative example. do not need to worry about query_node appears to be the child of anchor_node
            if self.graph.out_degree(anchor_node) <= self.expand_factor:
                siblings = [edge[1] for edge in self.graph.out_edges(anchor_node)]
            else:
                siblings = [edge[1] for edge in random.choices(list(self.graph.out_edges(anchor_node)), k=self.expand_factor)]
        else:  # positive example. remove query_node from the children set of anchor_node
            if self.graph.out_degree(anchor_node) <= self.expand_factor:
                siblings = [edge[1] for edge in self.graph.out_edges(anchor_node) if edge[1] != query_node]
            else:
                siblings = [edge[1] for edge in random.choices(list(self.graph.out_edges(anchor_node)), k=self.expand_factor) if edge[1] != query_node]
        nodes.extend(siblings)
        nodes_pos.extend([2]*len(siblings))

        # create dgl graph with features
        g = dgl.DGLGraph()
        g.add_nodes(len(nodes), {"x": self.node_features[nodes, :], "_id": torch.tensor(nodes), "pos": torch.tensor(nodes_pos)})
        g.add_edges(list(range(parent_node_idx)), parent_node_idx)
        g.add_edges(parent_node_idx, list(range(parent_node_idx+1, len(nodes))))

        # add self-cycle
        g.add_edges(g.nodes(), g.nodes())

        return g

