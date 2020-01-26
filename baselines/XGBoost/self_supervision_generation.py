"""
__author__: Jiaming Shen
__description__: Generate self supervision training/validation data
"""
import sys
sys.path.append("../../")
import numpy as np
import xgboost as xgb
from gensim.models import KeyedVectors
import torch.nn.functional as F
from data_loader.dataset import MAGDataset
import networkx as nx
from tqdm import tqdm
import argparse
from feature_extractor import NegativeQueue, FeatureExtractor

def main(args):
    graph_dataset = MAGDataset(name="", path=args.data, raw=False)

    node_features = graph_dataset.g_full.ndata['x']
    node_features = F.normalize(node_features, p=2, dim=1)
    vocab = graph_dataset.vocab
    full_graph = graph_dataset.g_full.to_networkx()
    kv = KeyedVectors(vector_size=node_features.shape[1])
    kv.add([str(i) for i in range(len(vocab))], node_features.numpy())

    if args.mode == "train":
        node_list = graph_dataset.train_node_ids
        graph = full_graph.subgraph(graph_dataset.train_node_ids).copy()
    elif args.mode == "validation":
        node_list = graph_dataset.validation_node_ids
        graph = full_graph.subgraph(graph_dataset.train_node_ids + graph_dataset.validation_node_ids).copy()
    else:
        node_list = graph_dataset.test_node_ids
        graph = full_graph.subgraph(graph_dataset.train_node_ids + graph_dataset.test_node_ids).copy()

    roots = [node for node in graph.nodes() if graph.in_degree(node) == 0]
    interested_node_set = set(node_list) - set(roots)
    node_list = list(interested_node_set)
    nq = NegativeQueue(node_list.copy()*5)  

    node2parents = {}  # list of correct parent positions
    node2masks = {}  # list of positions that should not be chosen as negative positions
    for node in tqdm(graph.nodes(), desc="generating intermediate data ..."):
        parents = [edge[0] for edge in graph.in_edges(node)]
        node2parents[node] = parents
        if node in interested_node_set:
            descendants = nx.descendants(graph, node)
            masks = set(list(descendants) + parents + [node] + roots)  
            node2masks[node] = masks

    edge_to_remove = []
    if args.mode == "validation":
        for node in graph_dataset.validation_node_ids:
            edge_to_remove.extend(list(graph.in_edges(node)))
        print(f"Remove {len(edge_to_remove)} edges between validation nodes and training nodes")
    graph.remove_edges_from(edge_to_remove)
    print("=== Finish data loading ===\n")

    feature_extractor = FeatureExtractor(graph, kv)
    NEGATIVE_RATIO = args.neg
    featMat = []
    labels = []
    for query_node in tqdm(node_list):
        cnt = 0
        for positive_parent in node2parents[query_node]:
            featMat.append(feature_extractor.extract_features(query_node, positive_parent))
            labels.append(1)
            cnt += 1
        
        num_negatives = NEGATIVE_RATIO * cnt
        avoid_set = node2masks[query_node]
        negatives = nq.sample_avoid_positive_set(avoid_set, num_negatives)
        for negative_parent in negatives:
            featMat.append(feature_extractor.extract_features(query_node, negative_parent))
            labels.append(0)
                
    data = xgb.DMatrix(np.array(featMat), label=np.array(labels), missing=-999.0)
    data.save_binary(args.output)

if __name__ == "__main__":
    # Example:
    #   --data: "/home/t-jishen/StructureExpan/data/MAG_FoS/computer_science.pickle.20190624.bin"
    #   --output: "/datadrive/structure_expan/data/MAG_FoS/mag_cs_train_1102.buffer"
    #   --mode: train
    #   --neg: 30
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', required=True, type=str, help='input file path')
    parser.add_argument('-o', '--output', required=True, type=str, help='output path of generated data')
    parser.add_argument('-m', '--mode', default="train", choices=["train", "validation"], type=str, help='mode of feature generator either train or validation, (default: train)')
    parser.add_argument('-n', '--neg', default=30, type=int, help='negative sampling ratio (default: 30)')
    args = parser.parse_args()
    print(f"Parameters: {args}")
    main(args)
