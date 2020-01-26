import sys
sys.path.append("../")
import torch
import torch.nn as nn
import dgl
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from data_loader.dataset import MAGDataset
from model.metric import calculate_ranks_from_distance, micro_mr, macro_mr, hit_at_1, hit_at_3, hit_at_5, mrr_scaled_10, combined_metrics
import time 

import argparse

FIRST_RUN = False 
NUM_PROCESSES = 10
# graph_bin_in = "../data/MAG_FoS/computer_science.pickle.20190624.bin"
# node_embed_txt_out = "../data/MAG_FoS/computer_science.pickle.20190624.vec.txt"
graph_bin_in = "/datadrive/structure_expan/data/MAG_FoS/mag_field_of_studies.pickle.20190702.bin"
node_embed_txt_out = "/datadrive/structure_expan/data/MAG_FoS/mag_field_of_studies.pickle.20190702.vec.txt"

def get_agg_func(mode="sum"):
    if mode == "sum":
        return np.sum
    elif mode == "mean":
        return np.mean
    elif mode == "min":
        return np.min
    elif mode == "max":
        return np.max

def get_one_query_rank(test_node_id, agg_func, beta, node_embeddings, idx2word, all_training_words, nxg, train_node_ids, train_node_ids_set,
                      node_idx2position_idx):
    distances = node_embeddings.distances(idx2word[test_node_id], all_training_words)
    parent_node_ids = set([edge[0] for edge in nxg.in_edges(test_node_id)])
    position_scores = []
    positive_relations = []
    total_lines = len(train_node_ids)
    count = 0.0
    old_percents = 0
    for position_idx, train_node_id in enumerate(train_node_ids):
        count += 1
        new_percents = count / total_lines
        if new_percents - old_percents > 0.001:
            sys.stdout.write("\rFinish %s percents" % str(new_percents * 100))
            sys.stdout.flush()
            old_percents = new_percents

        parent_distance = distances[position_idx]
        sibling_node_idx = [edge[1] for edge in nxg.out_edges(train_node_id) if edge[1] in train_node_ids_set]
        if len(sibling_node_idx) > 0:  # contain at least one sibling node
            sibling_node_position_idx = [node_idx2position_idx[node_idx] for node_idx in sibling_node_idx]
            sibling_distance = agg_func(distances[sibling_node_position_idx])
        else:
            sibling_distance = 0.0
        
        if beta == 0.0:
            position_scores.append( (parent_distance + sibling_distance) / (1+len(sibling_node_idx)) )
        else:
            position_scores.append(parent_distance + beta * sibling_distance)

        if train_node_id in parent_node_ids:
            positive_relations.append(position_idx)

    position_scores = np.array(position_scores)
    ranks = calculate_ranks_from_distance(position_scores, positive_relations)
    return ranks

def main(args):
    agg_func = get_agg_func(args.agg)
    mag_dataset = MAGDataset(name="computer_science", path=graph_bin_in, raw=False)
    if FIRST_RUN:
        print(f"First run, saving embedding to file {node_embed_txt_out}")
        node_feature = mag_dataset.g_full.ndata['x'].numpy()
        vocab = mag_dataset.vocab
        with open(f"{node_embed_txt_out}", "w") as fout:
            fout.write(f"{node_feature.shape[0]} {node_feature.shape[1]}\n")
            for row_id, row in enumerate(node_feature):
                word = "_".join(vocab[row_id].split())
                embed_string = " ".join([str(ele) for ele in row])
                fout.write(f"{word} {embed_string}\n")
    
    node_embeddings = KeyedVectors.load_word2vec_format(f"{node_embed_txt_out}")
    vocab = ["_".join(word.split()) for word in mag_dataset.vocab]
    idx2word = {k:v for k, v in enumerate(vocab)}

    nxg = mag_dataset.g_full.to_networkx()
    test_node_ids = mag_dataset.test_node_ids
    train_node_ids = mag_dataset.train_node_ids
    train_node_ids_set = set(train_node_ids)
    all_training_words = [idx2word[idx] for idx in train_node_ids]
    node_idx2position_idx = {node_idx:position_idx for position_idx, node_idx in enumerate(train_node_ids)}

    get_one_query_rank_test_node = partial(get_one_query_rank, agg_func=agg_func, beta=args.beta, node_embeddings=node_embeddings, idx2word=idx2word, 
                                        all_training_words=all_training_words, nxg=nxg, train_node_ids=train_node_ids, 
                                        train_node_ids_set=train_node_ids_set, node_idx2position_idx=node_idx2position_idx)

    # process all test_node_ids using multiprocess programming
    start = time.time()
    pool = Pool(processes = NUM_PROCESSES)
    all_ranks = pool.map(get_one_query_rank_test_node, test_node_ids)  # a list of rank positions 
    end = time.time()

    # print evaluation result
    print(f"Finished in {end-start} seconds")
    print(f"Number of test nodes: {len(test_node_ids)}")
    print(f"Number of candidate positions: {len(all_training_words)}")
    print(f"Macro-Averaged Mean Rank: {macro_mr(all_ranks)}")
    print(f"Hit@1: {hit_at_1(all_ranks)}")
    print(f"Hit@3: {hit_at_3(all_ranks)}")
    print(f"Hit@5: {hit_at_5(all_ranks)}")
    print(f"MRR (scale=10): {mrr_scaled_10(all_ranks)}")
    print(f"combined_metrics (scale=10): {combined_metrics(all_ranks)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agg', type=str, default='sum', choices=["mean", "sum", "min", "max"], help="sibling aggregation mode")
    parser.add_argument('--beta', type=float, default=0.0, help="balance factor between parent distance and aggregated sibling distances, 0.0 means direct aggregate between parent and siblings")
    args = parser.parse_args()
    if args.beta == 0.0:
        print("simple average between siblings and parents, discard their structural differences")
        assert args.agg == "sum", "when beta is 0.0, the aggregation mode must be sum"
    print(f"Parameters: {args}")
    main(args)
