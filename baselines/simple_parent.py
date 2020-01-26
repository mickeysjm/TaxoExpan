import sys
sys.path.append("../")
import torch
import torch.nn as nn
import dgl
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm

from data_loader.dataset import MAGDataset
from model.metric import calculate_ranks_from_distance, micro_mr, macro_mr, hit_at_1, hit_at_3, hit_at_5, mrr_scaled_10, combined_metrics

FIRST_RUN = True 
# graph_bin_in = "../data/MAG_FoS/computer_science.pickle.20190624.bin"
# node_embed_txt_out = "../data/MAG_FoS/computer_science.pickle.20190624.vec.txt"
graph_bin_in = "/datadrive/structure_expan/data/MAG_FoS/mag_field_of_studies.pickle.20190702.bin"
node_embed_txt_out = "/datadrive/structure_expan/data/MAG_FoS/mag_field_of_studies.pickle.20190702.vec.txt"

def main():
    mag_dataset = MAGDataset(name="computer_science", path=graph_bin_in, raw=False)
    if FIRST_RUN:
        print(f"First run, saving embedding to file {node_embed_txt_out}")
        node_feature = mag_dataset.g_full.ndata['x'].numpy()
        vocab = mag_dataset.vocab
        with open(node_embed_txt_out, "w") as fout:
            fout.write(f"{node_feature.shape[0]} {node_feature.shape[1]}\n")
            for row_id, row in enumerate(node_feature):
                word = "_".join(vocab[row_id].split())
                embed_string = " ".join([str(ele) for ele in row])
                fout.write(f"{word} {embed_string}\n")
    
    node_embeddings = KeyedVectors.load_word2vec_format(node_embed_txt_out)
    vocab = ["_".join(word.split()) for word in mag_dataset.vocab]
    idx2word = {k:v for k, v in enumerate(vocab)}

    nxg = mag_dataset.g_full.to_networkx()
    test_node_ids = mag_dataset.test_node_ids
    train_node_ids = mag_dataset.train_node_ids
    all_training_words = [idx2word[idx] for idx in train_node_ids]
    all_ranks = []  # a list of rank positions 
    for test_node_id in tqdm(test_node_ids):
        parent_node_ids = set([edge[0] for edge in nxg.in_edges(test_node_id)])
        distances = node_embeddings.distances(idx2word[test_node_id], all_training_words)
        positive_relations = [idx for idx, node_id in enumerate(train_node_ids) if node_id in parent_node_ids]
        ranks = calculate_ranks_from_distance(distances, positive_relations)
        all_ranks.append(ranks)
    
    # print evaluation result
    print(f"Number of test nodes: {len(test_node_ids)}")
    print(f"Number of candidate positions: {len(all_training_words)}")
    print(f"Macro-Averaged Mean Rank: {macro_mr(all_ranks)}")
    print(f"Hit@1: {hit_at_1(all_ranks)}")
    print(f"Hit@3: {hit_at_3(all_ranks)}")
    print(f"Hit@5: {hit_at_5(all_ranks)}")
    print(f"MRR (scale=10): {mrr_scaled_10(all_ranks)}")
    print(f"combined_metrics (scale=10): {combined_metrics(all_ranks)}")

if __name__ == "__main__":
    main()
