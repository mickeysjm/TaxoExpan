import torch
import torch.nn as nn 
import torch.nn.functional as F
from itertools import product
import numpy as np
import re


def nll_loss(output, target):
    return F.nll_loss(output, target) 

def square_exp_loss(output, target, beta=1.0):
    """
    output: a (batch_size, 1) tensor, value should be positive
    target: a (batch_size, ) tensor of dtype int
    beta: a float weight of negative samples
    """
    loss = (output[target==1]**2).sum() + beta * torch.exp(-1.0*output[target==0]).sum()
    return loss

def bce_loss(output, target, beta=1.0):
    """
    output: a (batch_size, 1) tensor
    target: a (batch_size, ) tensor of dtype int
    
    Note: here we revert the `target` because `output` is the "energy score" and thus smaller value indicates it is more likely to be a true position 
    """
    loss = F.binary_cross_entropy_with_logits(output.squeeze(), 1.0-target.float(), reduction="sum")
    return loss

def margin_rank_loss(output, target, margin=1.0):
    label = target.cpu().numpy()
    sep_01 = np.array([0, 1], dtype=label.dtype)
    sep_10 = np.array([1, 0], dtype=label.dtype)

    # fast way to find subarray indices in a large array, c.f. https://stackoverflow.com/questions/14890216/return-the-indexes-of-a-sub-array-in-an-array
    sep10_indices = [(m.start() // label.itemsize)+1 for m in re.finditer(sep_10.tostring(), label.tostring())]
    end_indices = [(m.start() // label.itemsize)+1 for m in re.finditer(sep_01.tostring(), label.tostring())]
    end_indices.append(len(label))
    start_indices = [0] + end_indices[:-1]

    pair_indices = []
    for start, middle, end in zip(start_indices, sep10_indices, end_indices):
        pair_indices.extend(list(product(range(start, middle), range(middle, end))))
    positive_indices = [ele[0] for ele in pair_indices]
    negative_indices = [ele[1] for ele in pair_indices]

    y = -1 * torch.ones(output[positive_indices,:].shape[0]).to(target.device)
    loss = F.margin_ranking_loss(output[positive_indices,:], output[negative_indices,:], y, margin=margin, reduction="sum")
    return loss

def info_nce_loss(output, target):
    """
    output: a (batch_size, 1+negative_size) tensor
    target: a (batch_size, ) tensor of dtype long, all zeros
    """
    return F.cross_entropy(output, target, reduction="sum")
