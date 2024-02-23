"""
Deformable attention implementation.
"""

import os
import numpy as np
import math

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import torch.autograd as autograd

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
        

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        assert self.emb_size % self.num_heads == 0, "Invalid head number!"

    def forward(self, x: Tensor, mask: Tensor = None, query: Tensor = None) -> Tensor:
        # print("*** MHA forward ***")
        if query != None:
            queries = rearrange(query, "b n (h d) -> b h n d", h=self.num_heads)
        else:
            queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        # print(att.shape)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
        
class DeformableTransformerEncoderBlock(nn.Module):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5,
                 num_of_points=10):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.drop_p = drop_p
        self.forward_expansion = forward_expansion
        self.forward_drop_p = forward_drop_p
        
        self.ln_1 = nn.LayerNorm(emb_size)
        self.dca = DeformableCrossAttention(num_heads, emb_size, drop_p, num_of_points)
        self.dropout1 = nn.Dropout(drop_p)
        
        self.ffc = ResidualAdd(nn.Sequential(
                        nn.LayerNorm(emb_size),
                        FeedForwardBlock(
                            emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                        nn.Dropout(drop_p)
                    ))
    
    def forward(self, input):
        x = input
        input = self.ln_1(input)
        # temp = input.detach()
        input = self.dca(input, input)
        input = self.dropout1(input) + x
        
        input = self.ffc(input)
        return input
        
class DeformableTransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, config=None):
        if config != None:
            super().__init__(*[DeformableTransformerEncoderBlock(emb_size, **config["encoder_config"])
                               for _ in range(depth)])
        else:
            super().__init__(*[DeformableTransformerEncoderBlock(emb_size) for _ in range(depth)])     


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, config=None):
        if config != None:
            super().__init__(*[TransformerEncoderBlock(emb_size, **config["encoder_config"]) for _ in range(depth)])
        else:
            super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])



class ClassificationHead(nn.Module):
    def __init__(self, emb_size=40, n_classes=4, config=None):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes

        hidden_size_1 = 256
        hidden_size_2 = 32
        drop_p_1 = 0.5
        drop_p_2 = 0.3
        if config != None:
            hidden_size_1 = config["hidden_size_1"]
            hidden_size_2 = config["hidden_size_2"]
            drop_p_1 = config["drop_p_1"]
            drop_p_2 = config["drop_p_2"]
        
        # self.classification_proj = nn.Sequential(
        #         nn.Linear(emb_size, hidden_size_1),
        #         nn.ELU(),
        #         nn.Dropout(drop_p_1),
        #         nn.Linear(hidden_size_1, hidden_size_2),
        #         nn.ELU(),
        #         nn.Dropout(drop_p_2),
        #         nn.Linear(hidden_size_2, 1)
        #     )
        
        self.classification_mlps = nn.ModuleList()
        for _ in range(n_classes):
            mlp = nn.Sequential(
                nn.Linear(emb_size, hidden_size_1),
                nn.ELU(),
                nn.Dropout(drop_p_1),
                nn.Linear(hidden_size_1, hidden_size_2),
                nn.ELU(),
                nn.Dropout(drop_p_2),
                nn.Linear(hidden_size_2, 2)
            )
            self.classification_mlps.append(mlp)

    def forward(self, input):
        '''
        Input: (batch_size, n_classes, emb_size)
        '''
        
        xs = torch.chunk(input, chunks=self.n_classes, dim=1) # (bs, 1, emb)
        
        # multiple mlps, binary classification
        outputs = []
        for i, mlp in enumerate(self.classification_mlps):
            # output = F.softmax(mlp(xs[i].squeeze(dim=1))) # (bs, 2): binary classification
            output = mlp(xs[i].squeeze(dim=1))
            outputs.append(output[:, 0].unsqueeze(1)) # the prob that query is the i-th class
        output = torch.cat(outputs, dim=1)
        
        
        # single mlp, multiple classification
        # use the first mlps
        # outputs = []
        # mlp = self.classification_mlp
        # for i in range(self.n_classes):
        #     # output = F.softmax(mlp(xs[i].squeeze(dim=1)))
        #     output = mlp(xs[i].squeeze(dim=1))
        #     outputs.append(output[:, i].unsqueeze(1))
        # output = torch.cat(outputs, dim=1)
        
        # use one mlp to do projection
        # output = self.classification_proj(input).squeeze()
       
        # strange good performance
        # outputs = []
        # for i, mlp in enumerate(self.classification_mlps):
        #     output = mlp(xs[i].squeeze(dim=1))
        #     output_max, _ = torch.max(output, dim=1, keepdim=True)
        #     outputs.append(output_max)
        # output = torch.cat(outputs, dim=1)
        return output


class DeformableCrossAttention(nn.Module):
    def __init__(self, num_heads, emb_size, drop_p=0.5, num_of_points=10):
        '''
        query: (bs, n_classes, emb_size)

        Use nn.Linear to get the reference points and weights.
        '''
        super().__init__()
        self.drop_p = drop_p
        self.num_of_points = num_of_points

        self.fc_pts = nn.Linear(emb_size, num_of_points)
        self.fc_w = nn.Linear(emb_size, num_of_points)
        self.att = MultiHeadAttention(emb_size, num_heads, drop_p)
        

    def forward(self, input, query):
        # print("*** DCA forward ***")
        bs, n, e = input.shape
        ref_pts_idx = self.fc_pts(query) # (bs, n_classes, num_of_points) point offset
        ref_pts_idx = torch.floor(
            torch.sigmoid(ref_pts_idx) * n
        ).long() # int [0, n-1]

        ref_weight = self.fc_w(query) # (bs, n_classes, num_of_points) point weight
        ref_weight = F.softmax(ref_weight, -1) # float [0,1]

        indices_tuple = ref_pts_idx.split(1, dim=1)
        indices_lists = [t.squeeze() for t in indices_tuple] # list of idx tensor (bs, num_of_points)

        weight_lists = ref_weight.split(1, dim=1)
        weight_lists = [t.squeeze() for t in weight_lists] # list of tensor

        # list of (bs, num_of_points, e) tensors with weight multiplied
        deform_tensors = []
        for i in range(len(indices_lists)):
            index = indices_lists[i].unsqueeze(-1).repeat(1,1,e)
            tmp_t = input.gather(1, index) # (bs, num_of_points, e)
            weights_tensor = weight_lists[i].unsqueeze(-1).repeat(1,1,e) # (bs, num_of_points, e)
            # print(weight_lists[i].shape)
            # print(tmp_t.shape)
            # print(weights_tensor.shape)
            deform_tensors.append(tmp_t * weights_tensor)

        att_ans_list = []
        for t in deform_tensors:
            att = self.att(x=t, mask=None, query=query) # (bs, num_of_points, e)
            att = torch.sum(att, dim=1) # (bs, e)
            att_ans_list.append(att)
        ans = torch.stack(att_ans_list, dim=1)


        # # for long seq: add first, one attention
        # deform_tensors = []
        # for i in range(len(indices_lists)):
        #     index = indices_lists[i].unsqueeze(-1).repeat(1,1,e)
        #     tmp_t = input.gather(1, index) # (bs, num_of_points, e)
        #     weights_tensor = weight_lists[i].unsqueeze(-1).repeat(1,1,e) # (bs, num_of_points, e)
        #     weighted_tensor = tmp_t * weights_tensor
        #     deform_tensors.append(weighted_tensor.sum(dim=1))
        # deform_tensor = torch.stack(deform_tensors, dim=1)
        # ans = self.att(x=deform_tensor, mask=None, query=query)
        
        return ans
    
        
        
class TransformerDecoderBlock(nn.Module):
    def __init__(self, emb_size, 
                num_heads=10, 
                drop_p=0.5, 
                forward_expansion=4, 
                forward_drop_p=0.5,
                num_of_points=10):
        '''
        n_classes == num of object queries
        '''
        super().__init__()
        self.p1 = ResidualAdd(nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p)
                ))
        # self.p2 = 
        self.ln = nn.LayerNorm(emb_size)
        self.deform_cross_att = DeformableCrossAttention(num_heads, emb_size, drop_p, num_of_points)
        # self.cross_att = MultiHeadAttention(emb_size, num_heads, drop_p)
        self.dropout = nn.Dropout(drop_p)
        
        self.p3 = ResidualAdd(nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p)
                ))

    def forward(self, feature, query):
        '''
        feature: (bs, n, emb)
        query: (bs, n_classes, emb)
        '''
        
        query = self.p1(query)
        query = self.ln(query)
        att = self.deform_cross_att(feature, query)
        # att = self.cross_att(query, mask=None, query=feature) # transformer decoder
        # print("feature: ", feature.shape)
        # print("query: ", query.shape)
        att = self.dropout(att)
        att = self.p3(att)
        return att

class TransformerDecoder(nn.Module):
    def __init__(self, depth, n_classes=4, emb_size=40, config=None):
        super().__init__()
        self.depth = depth
        self.decoder_blocks = nn.ModuleList()
        for i in range(depth):
            self.decoder_blocks.append(TransformerDecoderBlock(emb_size, **config["decoder_config"]))

        # try linearly project feature to object queries
        # hard code 61 here: a potential bug
        # self.obj_query_proj = nn.Linear(61, n_classes)
        
        # randomly initialize query
        self.obj_query = nn.Parameter(torch.randn(n_classes, emb_size))

        '''
        note: two ways of query initialization hardly influence performance.
        '''
        
    def forward(self, input):
        '''
        input: (bs, n, emb)
        '''
        bs, n, emb = input.shape
        # print(input.shape)
        
        # batch_query = self.obj_query_proj(input.permute(0,2,1)).permute(0,2,1)
        batch_query = self.obj_query.unsqueeze(0).repeat(bs, 1, 1) #(bs, n_cls, e)
        for i in range(self.depth):
            batch_query = self.decoder_blocks[i](input, batch_query)
        return batch_query
    
