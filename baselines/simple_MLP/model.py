import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import dgl.function as fn
import itertools
import re

### Model ###
class MLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, first_hidden, second_hidden, activation, pretrained_embedding=None):
        super(MLP, self).__init__()
        if type(pretrained_embedding) != type(None):
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
            self.embed.weight.requires_grad = False
        else:
            self.embed = nn.Embedding(vocab_size, embed_dim)
        
        self.classify = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embed_dim*2, first_hidden),
            activation,
            nn.Linear(first_hidden, second_hidden),
            activation,
            nn.Linear(second_hidden, 1)
        )
        
    def forward(self, parents, children):
        return self.classify(torch.cat([self.embed(parents), self.embed(children)], dim=1))


class DeepSetMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, first_hidden, second_hidden, activation, pretrained_embedding=None):
        super(DeepSetMLP, self).__init__()
        if type(pretrained_embedding) != type(None):
            zero_vector = np.zeros([1, embed_dim], dtype=pretrained_embedding.dtype)
            pretrained_embedding = np.vstack((pretrained_embedding, zero_vector))
            self.embed = nn.Embedding(vocab_size+1, embed_dim)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
            self.embed.weight.requires_grad = False
        else:
            self.embed = nn.Embedding(vocab_size+1, embed_dim)
        
        self.sibling_dropout = nn.Dropout(0.5)

        self.set_encoder = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            activation,
            nn.Linear(embed_dim*2, embed_dim)
        )

        self.classify = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embed_dim*3, first_hidden),
            activation,
            nn.Linear(first_hidden, second_hidden),
            activation,
            nn.Linear(second_hidden, 1)
        )
        
    def forward(self, parents, siblings, children):
        # Without DeepSet Encoding
        # sibling_set_embed = torch.sum(self.embed(siblings), dim=1)
        
        # With DeepSet Encoding
        sibling_embed = torch.sum(self.sibling_dropout(self.embed(siblings)), dim=1)
        sibling_set_embed = self.set_encoder(sibling_embed)
        return self.classify(torch.cat([self.embed(parents), self.embed(children), sibling_set_embed], dim=1))

class DeepAPGMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, first_hidden, second_hidden, activation, pretrained_embedding=None):
        super(DeepAPGMLP, self).__init__()
        if type(pretrained_embedding) != type(None):
            zero_vector = np.zeros([1, embed_dim], dtype=pretrained_embedding.dtype)
            pretrained_embedding = np.vstack((pretrained_embedding, zero_vector))
            self.embed = nn.Embedding(vocab_size+1, embed_dim)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
            self.embed.weight.requires_grad = False
        else:
            self.embed = nn.Embedding(vocab_size+1, embed_dim)
        
        # sibling deepset
        self.sibling_dropout = nn.Dropout(0.5)
        self.sibling_set_encoder = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            activation,
            nn.Linear(embed_dim*2, embed_dim)
        )

        # grandparent deepset
        self.grand_parent_dropout = nn.Dropout(0.5)
        self.grand_parent_set_encoder = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            activation,
            nn.Linear(embed_dim*2, embed_dim)
        )

        self.classify = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embed_dim*4, first_hidden),
            activation,
            nn.Linear(first_hidden, second_hidden),
            activation,
            nn.Linear(second_hidden, 1)
        )
        
    def forward(self, parents, siblings, grand_parents, children):
        sibling_embed = torch.sum(self.sibling_dropout(self.embed(siblings)), dim=1)
        sibling_set_embed = self.sibling_set_encoder(sibling_embed)

        grand_parent_embed = torch.sum(self.sibling_dropout(self.embed(grand_parents)), dim=1)
        grand_parent_set_embed = self.grand_parent_set_encoder(grand_parent_embed)
        return self.classify(torch.cat([self.embed(parents), self.embed(children), sibling_set_embed, grand_parent_set_embed], dim=1))


### Training Loss ###
def bce_loss(output, target, beta=1.0):
    """
    output: a (batch_size, 1) tensor
    target: a (batch_size, ) tensor of dtype int
    """
    loss = F.binary_cross_entropy_with_logits(output.squeeze_(), 1.0-target.float(), reduction="sum")
    return loss


### Evaluation Metrics ###
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

def macro_mr(all_ranks):
    micro_mr = np.array([np.array(all_rank).mean() for all_rank in all_ranks]).mean()
    return micro_mr

def topk_hit(all_ranks, k=3):
    if isinstance(k, float):
        k = int(k*len(all_ranks))
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= k)
    return 1.0 * hits / len(rank_positions)

def scaled_MRR(all_ranks, scale=10):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    scaled_rank_positions = np.ceil(rank_positions / scale)
    return (1.0 / scaled_rank_positions).mean()

def obtain_ranks(outputs, targets):
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
        ranks = calculate_ranks_from_distance(distances, positive_relations)
        all_ranks.append(ranks)
    return all_ranks

def macro_averaged_rank(outputs, targets):
    all_ranks = obtain_ranks(outputs, targets)
    return macro_mr(all_ranks)

def batched_topk_hit_1(outputs, targets, k=1):
    all_ranks = obtain_ranks(outputs, targets)
    return topk_hit(all_ranks, k=k)

def batched_topk_hit_3(outputs, targets, k=3):
    all_ranks = obtain_ranks(outputs, targets)
    return topk_hit(all_ranks, k=k)

def batched_topk_hit_5(outputs, targets, k=5):
    all_ranks = obtain_ranks(outputs, targets)
    return topk_hit(all_ranks, k=k)

def batched_scaled_MRR(outputs, targets, scale=10):
    all_ranks = obtain_ranks(outputs, targets)
    return scaled_MRR(all_ranks, scale=scale)
