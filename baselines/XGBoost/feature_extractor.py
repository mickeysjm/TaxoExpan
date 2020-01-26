"""
__author__: Jiaming Shen
__description__: Feature extractor with human-designed features
"""
import numpy as np
from gensim.models import KeyedVectors
import networkx as nx
import random


class NegativeQueue(object):
    def __init__(self, queue):
        self.pointer = 0
        self.queue = queue
        random.shuffle(queue)

    def sample(self, query, negative_size):
        if self.pointer == 0:
            random.shuffle(self.queue)
        
        negatives = [ele for ele in self.queue[self.pointer: self.pointer+negative_size] if ele != query]

        self.pointer += negative_size
        if self.pointer >= len(self.queue):
            self.pointer = 0

        return negatives

    def sample_avoid_positive_set(self, positive_set, negative_size):
        """ positive_set: a set of elements should not be sampled """
        if self.pointer == 0:
            random.shuffle(self.queue)
        
        negatives = [ele for ele in self.queue[self.pointer: self.pointer+negative_size] if ele not in positive_set]

        self.pointer += negative_size
        if self.pointer >= len(self.queue):
            self.pointer = 0

        return negatives


class FeatureExtractor(object):
    def __init__(self, graph, kv):
        """ Conduct feature extraction
        
        Parameters
        ----------
        graph : networkx graph
            input full taxonomy graph
        kv : gensim.KeyedVector
            node embedding
        """
        self.graph = graph
        self.kv = kv
        
    def extract_features(self, query_node, parent_node):
        """
        query_node: a node index
        parent_node: the parent node index of query node

        return: feature vector of <query_node, parent_node> pair
        """
        neighbor = []
        neighbor.append(parent_node)

        grand_parents = [edge[0] for edge in self.graph.in_edges(parent_node)]
        num_gp = len(grand_parents)
        neighbor.extend(grand_parents)

        siblings = [edge[1] for edge in self.graph.out_edges(parent_node) if edge[1] != query_node]
        neighbor.extend(siblings)

        # calculate embedding distances
        distances = self.kv.distances(str(query_node), [str(ele) for ele in neighbor])
        p_distances = self.kv.distances(str(parent_node), [str(ele) for ele in neighbor])

        """
        calculate features
        """
        feat = []
        parent_distance = distances[0]
        # feature 1: query distance to parent node
        feat.append(parent_distance)

        # feature 2-9: query distance to grand parent node(s)
        grand_parent_distances = distances[1:1+num_gp]
        if len(grand_parent_distances) > 0:
            cnt_gp = len(grand_parent_distances)
            min_gp = np.min(grand_parent_distances)
            max_gp = np.max(grand_parent_distances)
            mid_gp = np.median(grand_parent_distances)
            avg_gp = np.mean(grand_parent_distances)
            std_gp = np.std(grand_parent_distances)
            ptp_gp = max_gp - min_gp
            parent_relative_rank = (grand_parent_distances < parent_distance).sum() / cnt_gp
            feat.extend([cnt_gp, min_gp, max_gp, mid_gp, avg_gp, std_gp, ptp_gp, parent_relative_rank])
        else:
            feat.extend([0, -999, -999, -999, -999, -999, -999, -999])

        # feature 10-17: query distance to sibling node(s)
        sibling_distances = distances[1+num_gp:]
        if len(sibling_distances) > 0:
            cnt_sib = len(sibling_distances)
            min_sib = np.min(sibling_distances)
            max_sib = np.max(sibling_distances)
            mid_sib = np.median(sibling_distances)
            avg_sib = np.mean(sibling_distances)
            std_sib = np.std(sibling_distances)
            ptp_sib = max_sib - min_sib
            sibling_relative_rank = (sibling_distances < parent_distance).sum() / cnt_sib
            feat.extend([cnt_sib, min_sib, max_sib, mid_sib, avg_sib, std_sib, ptp_sib, sibling_relative_rank])
        else:
            feat.extend([0, -999, -999, -999, -999, -999, -999, -999])

        # feature 18-25: parent distance to grand parent distance(s)
        parent_to_grand_parent_distances = p_distances[1:1+num_gp]
        if len(parent_to_grand_parent_distances) > 0:
            cnt_pgp = len(parent_to_grand_parent_distances)
            min_pgp = np.min(parent_to_grand_parent_distances)
            max_pgp = np.max(parent_to_grand_parent_distances)
            mid_pgp = np.median(parent_to_grand_parent_distances)
            avg_pgp = np.mean(parent_to_grand_parent_distances)
            std_pgp = np.std(parent_to_grand_parent_distances)
            ptp_pgp = max_pgp - min_pgp
            relative_rank = (parent_to_grand_parent_distances < parent_distance).sum() / cnt_pgp
            feat.extend([cnt_pgp, min_pgp, max_pgp, mid_pgp, avg_pgp, std_pgp, ptp_pgp, relative_rank])
        else:
            feat.extend([0, -999, -999, -999, -999, -999, -999, -999])

        # feature 26-33: parent distance to sibling distance(s)
        parent_to_sibling_distances = p_distances[1+num_gp:]
        if len(parent_to_sibling_distances) > 0:
            cnt_psib = len(parent_to_sibling_distances)
            min_psib = np.min(parent_to_sibling_distances)
            max_psib = np.max(parent_to_sibling_distances)
            mid_psib = np.median(parent_to_sibling_distances)
            avg_psib = np.mean(parent_to_sibling_distances)
            std_psib = np.std(parent_to_sibling_distances)
            ptp_psib = max_psib - min_psib
            relative_rank = (parent_to_sibling_distances < parent_distance).sum() / cnt_psib
            feat.extend([cnt_psib, min_psib, max_psib, mid_psib, avg_psib, std_psib, ptp_psib, relative_rank])
        else:
            feat.extend([0, -999, -999, -999, -999, -999, -999, -999])

        # feature 34-39: query to all local subgraph distances
        min_all = np.min(distances)
        max_all = np.max(distances)
        mid_all = np.median(distances)
        avg_all = np.mean(distances)
        std_all = np.std(distances)
        ptp_all = max_all - min_all
        feat.extend([min_all, max_all, mid_all, avg_all, std_all, ptp_all])
        
        return feat

    def extract_features_fast(self, query_node, parent_node, ego2parent, ego2children, parent_node_info, all_distances, tx_id2rank_id, rank_id2tx_id):
        """ Optimized version of feature extraction used in the prediction stage. 

        More information are cached to reduce repetitive computation. c.f. model_prediction.py script for details

        query_node: a node index
        parent_node: the parent node index of query node

        return: feature vector of <query_node, parent_node> pair
        """

        neighbor = []
        neighbor.append(parent_node)

        grand_parents = ego2parent[parent_node]
        num_gp = len(grand_parents)
        neighbor += grand_parents

        siblings = ego2children[parent_node]
        neighbor += siblings

        # calculate embedding distances
        # distances = kv.distances(str(query_node), [str(ele) for ele in neighbor])
        # p_distances = kv.distances(str(parent_node), [str(ele) for ele in neighbor])
        distances = all_distances[np.array([tx_id2rank_id[tx_id] for tx_id in neighbor])]
        p_distances = parent_node_info['p_distances']    
        
        """
        calculate features
        """
        feat = []
        parent_distance = distances[0]
        # feature 1: query distance to parent node
        feat.append(parent_distance)

        # feature 2-9: query distance to grand parent node(s)
        grand_parent_distances = distances[1:1+num_gp]
        if len(grand_parent_distances) > 0:
            cnt_gp = len(grand_parent_distances)
            min_gp = np.min(grand_parent_distances)
            max_gp = np.max(grand_parent_distances)
            mid_gp = np.median(grand_parent_distances)
            avg_gp = np.mean(grand_parent_distances)
            std_gp = np.std(grand_parent_distances)
            ptp_gp = max_gp - min_gp
            parent_relative_rank = (grand_parent_distances < parent_distance).sum() / cnt_gp
            feat.extend([cnt_gp, min_gp, max_gp, mid_gp, avg_gp, std_gp, ptp_gp, parent_relative_rank])
        else:
            feat.extend([0, -999, -999, -999, -999, -999, -999, -999])

        # feature 10-17: query distance to sibling node(s)
        sibling_distances = distances[1+num_gp:]
        if len(sibling_distances) > 0:
            cnt_sib = len(sibling_distances)
            min_sib = np.min(sibling_distances)
            max_sib = np.max(sibling_distances)
            mid_sib = np.median(sibling_distances)
            avg_sib = np.mean(sibling_distances)
            std_sib = np.std(sibling_distances)
            ptp_sib = max_sib - min_sib
            sibling_relative_rank = (sibling_distances < parent_distance).sum() / cnt_sib
            feat.extend([cnt_sib, min_sib, max_sib, mid_sib, avg_sib, std_sib, ptp_sib, sibling_relative_rank])
        else:
            feat.extend([0, -999, -999, -999, -999, -999, -999, -999])

        # feature 18-25: parent distance to grand parent distance(s)
        parent_to_grand_parent_distances = p_distances[1:1+num_gp]
        if len(parent_to_grand_parent_distances) > 0:
            cnt_pgp = len(parent_to_grand_parent_distances)
            min_pgp = np.min(parent_to_grand_parent_distances)
            max_pgp = np.max(parent_to_grand_parent_distances)
            mid_pgp = np.median(parent_to_grand_parent_distances)
            avg_pgp = np.mean(parent_to_grand_parent_distances)
            std_pgp = np.std(parent_to_grand_parent_distances)
            ptp_pgp = max_pgp - min_pgp
            relative_rank = (parent_to_grand_parent_distances < parent_distance).sum() / cnt_pgp
            feat.extend([cnt_pgp, min_pgp, max_pgp, mid_pgp, avg_pgp, std_pgp, ptp_pgp, relative_rank])
        else:
            feat.extend([0, -999, -999, -999, -999, -999, -999, -999])

        # feature 26-33: parent distance to sibling distance(s)
        parent_to_sibling_distances = p_distances[1+num_gp:]
        if len(parent_to_sibling_distances) > 0:
            cnt_psib = len(parent_to_sibling_distances)
            min_psib = np.min(parent_to_sibling_distances)
            max_psib = np.max(parent_to_sibling_distances)
            mid_psib = np.median(parent_to_sibling_distances)
            avg_psib = np.mean(parent_to_sibling_distances)
            std_psib = np.std(parent_to_sibling_distances)
            ptp_psib = max_psib - min_psib
            relative_rank = (parent_to_sibling_distances < parent_distance).sum() / cnt_psib
            feat.extend([cnt_psib, min_psib, max_psib, mid_psib, avg_psib, std_psib, ptp_psib, relative_rank])
        else:
            feat.extend([0, -999, -999, -999, -999, -999, -999, -999])

        # feature 34-39: query to all local subgraph distances
        min_all = np.min(distances)
        max_all = np.max(distances)
        mid_all = np.median(distances)
        avg_all = np.mean(distances)
        std_all = np.std(distances)
        ptp_all = max_all - min_all
        feat.extend([min_all, max_all, mid_all, avg_all, std_all, ptp_all])
        
        return feat
