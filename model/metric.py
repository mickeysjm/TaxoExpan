import torch
import numpy as np
import itertools
import re


def calculate_ranks_from_similarities(all_similarities, positive_relations):
    """
    all_similarities: a np array
    positive_relations: a list of array indices

    return a list
    """
    positive_relation_similarities = all_similarities[positive_relations]
    negative_relation_similarities = np.ma.array(all_similarities, mask=False)
    negative_relation_similarities.mask[positive_relations] = True
    ranks = list((negative_relation_similarities > positive_relation_similarities[:, np.newaxis]).sum(axis=1) + 1)
    return ranks

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

def obtain_ranks(outputs, targets, mode=0):
    """ 
    outputs : tensor of size (batch_size, 1), required_grad = False, model predictions
    targets : tensor of size (batch_size, ), required_grad = False, labels
        Assume to be of format [1, 0, ..., 0, 1, 0, ..., 0, ..., 0]
    mode == 0: rank from distance (smaller is preferred)
    mode == 1: rank from similarity (larger is preferred)
    """
    if mode == 0:
        calculate_ranks = calculate_ranks_from_distance
    else:
        calculate_ranks = calculate_ranks_from_similarities
    all_ranks = []
    prediction = outputs.cpu().numpy().squeeze()
    label = targets.cpu().numpy()
    sep = np.array([0, 1], dtype=label.dtype)
    
    # fast way to find subarray indices in a large array, c.f. https://stackoverflow.com/questions/14890216/return-the-indexes-of-a-sub-array-in-an-array
    end_indices = [(m.start() // label.itemsize)+1 for m in re.finditer(sep.tostring(), label.tostring())]
    end_indices.append(len(label)+1)
    start_indices = [0] + end_indices[:-1]
    for start_idx, end_idx in zip(start_indices, end_indices):
        distances = prediction[start_idx: end_idx]
        labels = label[start_idx:end_idx]
        positive_relations = list(np.where(labels == 1)[0])
        ranks = calculate_ranks(distances, positive_relations)
        all_ranks.append(ranks)
    return all_ranks

def macro_mr(all_ranks):
    macro_mr = np.array([np.array(all_rank).mean() for all_rank in all_ranks]).mean()
    return macro_mr

def micro_mr(all_ranks):
    micro_mr = np.array(list(itertools.chain(*all_ranks))).mean()
    return micro_mr

def hit_at_1(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 1)
    return 1.0 * hits / len(rank_positions)

def hit_at_3(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 3)
    return 1.0 * hits / len(rank_positions)

def hit_at_5(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 5)
    return 1.0 * hits / len(rank_positions)

def mrr_scaled_10(all_ranks):
    """ Scaled MRR score, check eq. (2) in the PinSAGE paper: https://arxiv.org/pdf/1806.01973.pdf
    """
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    scaled_rank_positions = np.ceil(rank_positions / 10)
    return (1.0 / scaled_rank_positions).mean()

def combined_metrics(all_ranks):
    """ 
    combination of three metrics, used in early stop 
    """
    score =  macro_mr(all_ranks) * (1.0/max(mrr_scaled_10(all_ranks), 0.0001)) * (1.0/max(hit_at_3(all_ranks), 0.0001)) * (1.0/max(hit_at_1(all_ranks), 0.0001)) 
    return score