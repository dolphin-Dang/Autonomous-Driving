import numpy as np
import torch
import torch.nn as nn

"""
实现一个简化版BEVFormer类及其辅助函数，仅仅用于加深理解。
并不搭建完整的训练逻辑。
并不实现所有函数细节，有些地方用注释和函数名表示函数内容。
"""

config = {
    "batch_size": 64,
    "queue": 4, # t-3, t-2, t-1, t
    "camera": 6,
    "output_size": 200,
    "embd_size": 256,
    "hidden_size": 128,
    "num_block": 6,
    "n_heads": 8,
    "dropout": 0.5,
    "num_ref_points": 4
}

def LossFunc(ground_truth, BEV_feature):
    """
    Input:
        ground_truth: dict={
            "3D_bbox": tuple(x,y,z,width,length,height,yaw angle),
            "label": class
        }
        BEV_feature: output of BEVFormer.forward()

    Given the input, calculate the loss of 
    detection and segmentation heads.
    """
    # TODO
    return "loss"

def get_reference_points(bev_query, num_ref_points, mode):
    """
    Input:
        bev_query: Tensor shape == (config["output_size"]^2, config["embd_size"])
        num_ref_points: number of ref pts to generate
        mode: "temporal" or "spatial"

    Take BEV query and num of reference points as input,
    output a nested list 
        "temporal" mode: 
            [
                [
                    [ref_pt1, ref_pt2, ..., ref_ptn], 
                    ......
                ], # seq_len
                ......
            ] # bs
            # each ref_pt is [i, j, w] indicating the offset(i,j) and weight w.

            where:
                bs: len(output)==bev_query.shape[0]
                seq_len: len(output[0])==bev_query.shape[1]
                n_ref: len(output[0][0])==num_ref_points
                2D coordinate: len(output[0][0][0])==3

        "spatial" mode:
            [
                [
                    [
                        [ref_pt1, ref_pt2, ..., ref_ptn],
                        ......
                    ], # seq_len
                    ......
                ], # num of cameras (view)
                ......
            ] # bs
            
            where:
                bs: len(output)==bev_query.shape[0]
                n_can: len(output[0])==config["camera"]
                seq_len: len(output[0][0])==bev_query.shape[1]
                n_ref: len(output[0][0][0])==num_ref_points
                2D coordinate: len(output[0][0][0][0])==3
    """


    # only generate [0,0]s here to be simple
    # actually we need a fullyconnected network here
    # TODO: generate real ref_pts
    reference_points = [[[[0,0] for _ in range (num_ref_points)]] * bev_query.shape[1]] * bev_query.shape[0]
    return reference_points

def do_attention(q, k, v):
    # let T = n_ref*seq_len or n_cam*n_ref*seq_len
    # (bs, T, nh, hs).permute(0,2,1,3) -> (bs, nh, T, hs)
    q = q.permute(0,2,1,3)
    k = k.permute(0,2,1,3)
    v = v.permute(0,2,1,3)

    # ( q @ k.T / sqrt(hs) ) * v
    # (bs, nh, T, hs) @ (bs, nh, hs, T) = (bs, nh, T, T)
    att = torch.matmul(q, k.T)
    att = att / sqrt(v.shape[-1])
    att = torch.softmax(att, dim=-1)
    att = self.dropout(att)
    # (bs, nh, T, T) @ (bs, nh, T, hs) = (bs, nh, T, hs)
    att = torch.matmul(att, v)
    return att


class TemporalSelfAttention(nn.Module):
    def __init__(self):
        super(TemporalSelfAttention, self).__init__()
        self.v = nn.Linear(
            config["batch_size"]*(config["output_size"]**2)*config["num_ref_points"], 
            config["embd_size"]
            )
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, history_bev, bev_query):
        """
        Input:
            history_bev, bev_query: shape == (bs, config["output_size"]^2, config["embd_size"])
        Output:
            Tensor: the same shape

        Do deformable-attention here.
        """

        bs, seq_len, embd_size = bev_query.shape
        n_heads = config["n_heads"]
        n_ref = config["num_ref_points"]

        assert(embd_size % n_heads == 0)
        q = bev_query.view(bs, seq_len, n_heads, embd_size//n_heads)
        k = history_bev.view(bs, seq_len, n_heads, embd_size//n_heads)
        v = v.view(bs, seq_len*n_ref, n_heads, embd_size//n_heads)

        ref_pts = get_reference_points(bev_query, config["num_ref_points"], mode="temporal")

        # get k(history_bev) change to (bs, n_ref*seq_len, nh, hs)
        # where every n_ref in seq_len have different value according to ref_pts
        # indicating the reference points in Deformable-Attention
        def bev_query_deform(k, ref_pts):
            """
            Input:
                k: bev_query, shape=(bs, seq_len, nh, hs)
                ref_pts: nested list of shape (bs, seq_len, n_ref, 3)
            Output:
                k': bev_query' of shape (bs, n_ref*seq_len, nh, hs)
            """
            # TODO: create k from ref_pts
            return k.repeat(1, len(ref_pts[0][0]), 1, 1)

        k = bev_query_deform(k, ref_pts)

        # get q(bev_query) change to (bs, n_ref*seq_len, nh, hs)
        # where every n_ref in seq_len have the same value (duplicate)
        # to match the ref_pts
        q = q.repeat(1, n_ref, 1, 1)

        att = do_attention(q, k, v)

        def get_BEV_feature_from_att(att, ref_pts):
            """
            Input:
                att: shape=(bs, nh, T=n_ref*seq_len, hs)
                ref_pts: nested list of shape (bs, seq_len, n_ref, 3)
                # need weight in ref_pts
            Output:
                bev_query: bev_query of shape (bs, seq_len, embd_size)            
            """
            # TODO: use weight to get att of shape (bs, seq_len, nh, hs)
            att = ...

            # reshape att to bev_query
            att = att.view(bs, seq_len, embd_size)
            return att

        return get_BEV_feature_from_att(att, ref_pts)
        

class SpatialCrossAttention(nn.Module):
    def __init__(self):
        super(SpatialCrossAttention, self).__init__()
        self.v = nn.Linear(
            config["batch_size"]*(config["output_size"]**2)*config["camera"], 
            config["embd_size"]
            )
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, single_frame_feature, bev_query):
        """
        Input:
            single_frame_feature: Tensor of size (bs, 1, cam, C', H', W')
            bev_query: Tensor of size (bs, output_size^2, embd_size)
        Output:
            bev_feature: Tensor of size (bs, output_size^2, embd_size)

        Do Spatial-Attention from n_cam views of single frame.
        """

        bs, seq_len, embd_size = bev_query.shape
        _, _, n_cam, _, _, _ = single_frame_feature.shape
        n_heads = config["n_heads"]
        n_ref = config["num_ref_points"]

        assert(n_cam == config["camera"])
        assert(embd_size % n_heads == 0)

        q = bev_query.view(bs, seq_len, n_heads, embd_size//n_heads)
        k = single_frame_feature.view(bs, n_cam, seq_len, n_heads, embd_size//n_heads)
        v = v.view(bs, seq_len*n_ref, n_heads, embd_size//n_heads)

        ref_pts = get_reference_points(bev_query, config["num_ref_points"], mode="spatial")

        # get k(history_bev) change to (bs, n_cam*_n_ref*seq_len, nh, hs)
        # where every n_cam*n_ref in seq_len have different value according to ref_pts
        # indicating the reference points in Deformable-Attention
        def bev_query_deform(k, ref_pts):
            """
            Input:
                k: bev_query, shape=(bs, seq_len, nh, hs)
                ref_pts: nested list of shape (bs, n_cam, seq_len, n_ref, 3)
            Output:
                k': bev_query' of shape (bs, n_cam*n_ref*seq_len, nh, hs)
            """
            # TODO: create k from ref_pts
            return k.repeat(1, len(ref_pts[0][0])*len(ref_pts[0][0][0]), 1, 1)

        k = bev_query_deform(k, ref_pts)

        # get q(bev_query) change to (bs, n_ref*seq_len, nh, hs)
        # where every n_ref in seq_len have the same value (duplicate)
        # to match the ref_pts
        q = q.repeat(1, n_cam*n_ref, 1, 1)

        att = do_attention(q, k, v)

        def get_BEV_feature_from_att(att, ref_pts):
            """
            Input:
                att: shape=(bs, nh, T=n_cam*n_ref*seq_len, hs)
                ref_pts: nested list of shape (bs, seq_len, n_ref, 3)
                # need weight in ref_pts
            Output:
                bev_query: bev_query of shape (bs, seq_len, embd_size)            
            """
            # TODO: use weight to get att of shape (bs, seq_len, nh, hs)
            att = ...

            # reshape att to bev_query
            att = att.view(bs, seq_len, embd_size)
            return att

        return get_BEV_feature_from_att(att, ref_pts)


class EncoderBlock(nn.Module):
    """
    An EncoderBlock have layers as follows:
        Temporal Self-Attention
        Add & Norm
        Spatial Cross-Attention
        Add & Norm
        Feed Forward
        Add & Norm

    All EncoderBlocks share a history BEV feature.
    Output feature from last Block is the input query of new Block.
    """
    def __init__(self):
        super(EncoderBlock, self).__init__()
        self.tsa = TemporalSelfAttention()
        self.sca = SpatialCrossAttention()
        self.mlp = nn.Sequential(
            nn.Linear(config["embd_size"], config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(config["hidden_size"], config["embd_size"])
        )
        self.ln_1 = nn.LayerNorm(config["embd_size"])
        self.ln_2 = nn.LayerNorm(config["embd_size"])
        self.ln_3 = nn.LayerNorm(config["embd_size"])

    def forward(self, history_bev, bev_query, single_frame_feature):
        """
        Input:
            history_bev: output of last BEVBlock. 
            bev_query: output of last EncoderBlock.
            single_frame_feature: Tensor of size (bs, 1, cam, C', H', W')
        Output:
            bev_feature.

        All Tensors above excluding single_frame_feature are of the same size (bs, output_size^2, embd_size).
        """
        bev_query = self.tsa(history_bev, bev_query) + bev_query
        bev_query = self.ln_1(bev_query)
        bev_query = self.sca(single_frame_feature, bev_query) + bev_query
        bev_query = self.ln_2(bev_query)
        bev_query = self.mlp(bev_query) + bev_query
        bev_query = self.ln_3(bev_query)
        return bev_query
        


class BEVBlock(nn.Module):
    """
    A BEVBlock have config["num_block"] EncoderBlocks.

    BEVBlock takes history BEV feature as input,
    and have its own BEV query.

    The first BEVBlock have no history BEV feature.
    """
    def __init__(self):
        super(BEVBlock, self).__init__()
        # BEV query: (output_size^2, embd_size)
        # learnable position embedding
        self.bev_query = nn.Embedding(config["output_size"]**2, config["embd_size"])
        self.EncoderBlock_list = [EncoderBlock() for _ in range(config["num_block"])]

    def forward(self, history_bev, single_frame_feature):
        """
        if history_bev == None:
            the first BEVBlock with no history, do self-Attention.
        else:
            do Temporal self-Attention.

        Input:
            history_bev: None or a Tensor of size (bs, config["output_size"]^2, config["embd_size"])
            single_frame_feature: Tensor of size (bs, 1, cam, C', H', W')
        Output:
            bev_feature: Tensor of size (bs, config["output_size"]^2, config["embd_size"])
        """

        bev_feature = self.bev_query.weight
        if history_bev == None:
            history_bev = bev_feature

        for i in range(len(self.EncoderBlock_list)):
            bev_feature = self.EncoderBlock_list[i](history_bev, bev_feature, single_frame_feature)
        return bev_feature


class BEVFormer(nn.Module):
    """
    Simplify toy BEVFormer class.
    Do not implement every method and details.

    BEVFormer:
        Input shape=(bs,q,cam,C,H,W),
        Output BEV feature map of size config["output_size"]^2.

    BEVFormer uses ground truth boxes and classes as labels.
    Let's say there's a loss function to measure the loss.

    A BEVFormer have config["queue"] BEVBlocks.
    """
    def __init__(self, backbone):
        """
        backbone:
            Backbone network to extract features from different views.
            Assume the backbone network takes (batch_size, C, H, W) as input.

            feature = self.backbone(img) # shape==(batch_size, C', H', W')

        Assume that the input shape matches with backbone,
        and the output of backbone meet the requirement: C' == config["embd_size"].
        """
        super(BEVFormer, self).__init__()
        self.backbone = backbone
        self.image_feature = None

        self.BEVBlock_list = [BEVBlock() for _ in range(config["queue"])]

    def forward(self, images):
        """
        Input:
            images: shape=(batch_size, 
                        queue, 
                        camera, 
                        C, H, W)
                where,  queue: consecutive frames
                        camera: pictures from different cameras
                        C: channel
                        H: height
                        W: width 

        Output:
            BEV feature map of size (bs, config["output_size"]^2, config["embd_size"]). Tensor.

        """

        backbone_input_data = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        self.image_feature = self.backbone(backbone_input_data) # (-1, C, H, W)
        self.image_feature = self.image_feature.view(
            *data.shape[:-3], 
            *self.image_feature.shape[-3:]
        ) # (bs, q, cam, C', H', W')
        assert(len(images.shape)==len(self.image_feature.shape))
        assert(images.shape[0]==self.image_feature.shape[0])
        assert(images.shape[1]==self.image_feature.shape[1])
        assert(images.shape[2]==self.image_feature.shape[2])

        bev_feature = None
        for i in range(len(self.BEVBlock_list)):
            single_frame_feature = self.image_feature[:, i, :] # (bs, 1, cam, C', H', W')
            bev_feature = self.BEVBlock_list[i](bev_feature, single_frame_feature)

        return bev_feature