"""
__author__: Jiaming Shen
__description__: Load a trained XGBoost model and perform prediction
"""
import sys
sys.path.append("../../")
import numpy as np
import pickle
import xgboost as xgb
from gensim.models import KeyedVectors
import torch.nn.functional as F
from data_loader.dataset import MAGDataset
import networkx as nx
from tqdm import tqdm 
import argparse
from feature_extractor import FeatureExtractor


def calculate_ranks_from_distance(all_distances, positive_relations):
    """
    all_distances: a np array
    positive_relations: a list of array indices

    return a list
    """
    positive_relation_distance = all_distances[positive_relations]
    negative_relation_distance = np.ma.array(all_distances, mask=False)
    negative_relation_distance.mask[positive_relations] = True
    ranks = list((negative_relation_distance < positive_relation_distance[:, np.newaxis]).sum(axis=1) + 1)
    return ranks


def main(args):
    """ Load graph dataset """
    graph_dataset = MAGDataset(name="", path=args.test_data, raw=False)
    node_features = graph_dataset.g_full.ndata['x']
    node_features = F.normalize(node_features, p=2, dim=1)
    vocab = graph_dataset.vocab
    full_graph = graph_dataset.g_full.to_networkx()
    kv = KeyedVectors(vector_size=node_features.shape[1])
    kv.add([str(i) for i in range(len(vocab))], node_features.numpy())

    node_list = graph_dataset.test_node_ids
    graph = full_graph.subgraph(graph_dataset.train_node_ids + graph_dataset.test_node_ids).copy()

    roots = [node for node in graph.nodes() if graph.in_degree(node) == 0]
    interested_node_set = set(node_list) - set(roots)
    node_list = list(interested_node_set)

    node2parents = {}  # list of correct parent positions
    node2masks = {}  # list of positions that should not be chosen as negative positions
    all_positions = set(graph_dataset.train_node_ids)  # each `position` is the candidate parent node of query element
    for node in tqdm(graph.nodes(), desc="generating intermediate data ..."):
        parents = [edge[0] for edge in graph.in_edges(node)]
        node2parents[node] = parents
        if node in interested_node_set:
            descendants = nx.descendants(graph, node)
            masks = set(list(descendants) + parents + [node] + roots)  
            node2masks[node] = masks

    edge_to_remove = []
    for node in graph_dataset.test_node_ids:
        edge_to_remove.extend(list(graph.in_edges(node)))
    print(f"Remove {len(edge_to_remove)} edges between test nodes and training nodes")
    graph.remove_edges_from(edge_to_remove)

    """ Cache information """
    all_positions_list = list(all_positions)
    tx_id2rank_id = {v: k for k, v in enumerate(all_positions_list)}
    rank_id2tx_id = {k: v for k, v in enumerate(all_positions_list)}
    all_positions_string = [str(ele) for ele in all_positions]

    parent_node2info = {}
    ego2parent = {}
    ego2children = {}
    for parent_node in tqdm(graph, desc="generate egonet distances"):
        neighbor = []
        neighbor.append(parent_node)

        grand_parents = [edge[0] for edge in graph.in_edges(parent_node)]
        ego2parent[parent_node] = grand_parents
        num_gp = len(grand_parents)
        neighbor.extend(grand_parents)

        siblings = [edge[1] for edge in graph.out_edges(parent_node)]
        ego2children[parent_node] = siblings
        neighbor.extend(siblings)

        # calculate embedding distances
        p_distances = kv.distances(str(parent_node), [str(ele) for ele in neighbor])
        parent_node2info[parent_node] = {
            "p_distances": p_distances,
            "num_gp": num_gp
        }

    """ Load model for prediction and save results """
    with open(args.model, "rb") as fin:
        model = pickle.load(fin)

    feature_extractor = FeatureExtractor(graph, kv)

    result_file_path = args.output_ranks
    with open(result_file_path, "w") as fout:
        for query_node in tqdm(node_list):
            featMat = []
            labels = list(range(len(node2parents[query_node])))
            # cache information
            all_distances = kv.distances(str(query_node), all_positions_string)

            # select negative 
            negative_pool = [str(ele) for ele in all_positions if ele not in node2masks[query_node]]
            negative_dist = kv.distances(str(query_node), negative_pool)
            if args.retrieval_size == -1:
                top_negatives = list(zip(negative_pool, negative_dist))
            else:        
                top_negatives = sorted(zip(negative_pool, negative_dist), key=lambda x:x[1])[:args.retrieval_size]
            negatives = [int(ele[0]) for ele in top_negatives]

            # add positive positions
            for positive_parent in node2parents[query_node]:
                parent_node_info = parent_node2info[positive_parent]
                features = feature_extractor.extract_features_fast(query_node, positive_parent, ego2parent, ego2children, parent_node_info, all_distances, tx_id2rank_id, rank_id2tx_id)
                featMat.append(features)

            # add negative positions
            for negative_parent in negatives:
                parent_node_info = parent_node2info[negative_parent]
                features = feature_extractor.extract_features_fast(query_node, negative_parent, ego2parent, ego2children, parent_node_info, all_distances, tx_id2rank_id, rank_id2tx_id)
                featMat.append(features)
                
            dtest = xgb.DMatrix(np.array(featMat), missing=-999)
            ypred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
            distance = 1.0 - ypred

            ranks = calculate_ranks_from_distance(distance, labels)
            fout.write(f"{ranks}\n")

if __name__ == "__main__":
    # Example:
    #   --model: "/datadrive/structure_expan/saved/models/XGBoost/mag_cs_1102.pickle"
    #   --test_data: "../../data/MAG_FoS/computer_science.pickle.20190624.bin"
    #   --output_ranks: "/home/t-jishen/StructureExpan/baselines/feature_based/mag_cs.ranks01.txt"
    #   --retrieval_size: -1
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, help='model pickle path')
    parser.add_argument('--test_data', required=True, type=str, help='test data file path')
    parser.add_argument('--output_ranks', required=True, type=str, help='output rank file path')
    parser.add_argument('--retrieval_size', default=-1, type=int, 
        help='number of candidate positions to be tested. If this value equals to -1, we will test all positive candidate \
            positions. Otherwise, we test only top retrieval_size candidates based on the embedding similarities')
    args = parser.parse_args()
    main(args)
