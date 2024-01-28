import numpy as np
import torch
import torch.nn as nn

"""
实现一个简化版BEVFormer类及其辅助函数，仅仅用于加深理解。
并不搭建完整的训练逻辑。
并不实现所有函数细节，有些地方用注释和函数名表示函数内容。
并不保证无bug。
"""

# config try to be the same as the paper (nuScenes data)
config = {
    "batch_size": 64,
    "queue": 3,
    "camera": 6,
    "output_size": 200
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

class BEVFormer(nn.Module):
    """
    Simplify toy BEVFormer class.
    Do not implement every method and details.

    BEVFormer:
        Input shape=(bs,q,cam,C,H,W),
        Output BEV feature map of size config["output_size"]^2.

    BEVFormer uses ground truth boxes and classes as labels.
    Let's say there's a loss function to measure the loss.

    """
    def __init__(self, data):
        """
        data: shape=(batch_size, 
                    queue, 
                    camera, 
                    C, H, W)
        where,  queue: consecutive frames
                camera: pictures from different cameras
                C: channel
                H: height
                W: width 
        """
        super(BEVFormer, self).__init__()
        self.origin_data = data


    def forward(self, x):
        """
        Output:
        BEV feature map of size config["output_size"]^2.
        """
        BEV_feature = None
        
        return BEV_feature