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
    "num_block": 6
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
    return "loss"


class TemporalSelfAttention(nn.Module):
    def __init__(self):
        super(TemporalSelfAttention, self).__init__()

    def forward(self, history_bev, bev_query):
        """
        Input:
            history_bev, bev_query: shape == (config["output_size"]^2, config["embd_size"])
        Output:
            Tensor: the same shape

        Do deformable-attention here.
        """
        pass

class SpatialCrossAttention(nn.Module):
    def __init__(self):
        super(SpatialCrossAttention, self).__init__()

    def forward(self, single_frame_feature, bev_query):
        pass


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
        Output:
            bev_feature.

        All Tensors above are of the same size (output_size^2, embd_size).
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
            history_bev: None or a Tensor.
        Output:
            bev_feature: Tensor of size (config["output_size"]^2, config["embd_size"])
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
            BEV feature map of size (config["output_size"]^2, config["embd_size"]). Tensor.

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