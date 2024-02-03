import torch
from torch import nn
from .toy_deformable_transformer import DeformableTransformer

config = {
    "batch_size": 64
}


class DeformableDETR(nn.Module):
    '''
    Multi-scale Deformable DETR.
    '''
    def __init__(self, n_class, hidden_size, backbone, N, n_feature_layer, deformable_transformer):
        super().__init__()
        '''
        Input:
            hidden_size:    embedding dim of transformer and the channel num of backbone output.
            backbone:       outputs multi-scale feature map.
                backbone.num_channels: List. 
                    List of num_channels of multi-scale feature map.
                    len = n_feature_layer
                backbone.feature_maps: List.
                    List of multi-scale feature maps.
                    len = n_feature_layer
            n_feature_layer:    num of feature layers the backbone outputs. 
                                For multi-scale detection.
            deformable_transformer: 
                Takes input:
                    multi-scale feature map with position embedding and layer embedding,
                    batch sized object query.
                Output:
                    tensor of same size as object query.

            N:  num of bbox DETR predicts.

        To simplify, assume that backbone outputs just n_feature_layer feature maps.
        A 1*1 conv is needed to project its channel nums to hidden_size.
        '''
        # self.n_class = n_class
        # self.hidden_size = hidden_size
        # self.N = N
        # self.n_feature_layer = n_feature_layer

        # Assume that the backbone output feature map 
        # that transformer block need
        self.backbone = backbone
        self.linear_class = nn.Linear(hidden_size, n_class+1) # background class
        self.linear_bbox = nn.Linear(hidden_size, 4) # (center_x, center_y, height, width)

        self.deformable_transformer = deformable_transformer

        # project feature map channels to hidden_size
        input_proj_list = []
        for i in range(n_feature_layer):
            in_channels = backbone.num_channels[i]
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_size, kernel_size=1, stride=1)
                )
            )
        self.input_proj = nn.ModuleList(input_proj_list)


        self.obj_query = nn.Parameter(torch.rand(N, hidden_size))
        # make sure K >= max(*featureMap.shape[-2:]) (i.e. H & W)
        # K = 50 in DETR src code
        # In deformable DETR src code, use sine and cosine embedding, we use learnable here.
        self.row_embd = nn.Parameter(torch.rand(K, hidden_size//2))
        self.col_embd = nn.Parameter(torch.rand(K, hidden_size//2))

        # In deformable DETR src code, use learnable layer embedding.
        self.layer_embd = nn.Parameter(torch.rand(n_feature_layer))

    def forward(self, input):
        '''
        Take fixed size picture as input, 
        output class, tuple of bbox.
        bbox = [0,1]^4, which means the scale on the original image.
        '''
        cnn_feature_list = self.backbone(input)  # List of (bs, ch, H, W)
        bs, ch, H, W = cnn_feature_list[0].shape 

        # position embedding
        row_repeat = self.row_embd[:H].unsqueeze(1).repeat(1, W, 1) # (H, W, hidden_size//2)
        col_repeat = self.col_embd[:W].unsqueeze(0).repeat(H, 1, 1) # (H, W, hidden_size//2)
        pos_embd = torch.cat([row_repeat, col_repeat], dim=-1).flatten(0, 1) # (H*W, hidden_size)
        pos_embd_batch = pos_embd.unsqueeze(0).repeat((config["batch_size"])) # (bs, H*W, hidden_size)

        # make all feature map hidden_size channel
        for i in range(len(cnn_feature_list)):
            cnn_feature_list[i] = self.input_proj[i](cnn_feature_list[i])
            cnn_feature_list[i] = cnn_feature_list[i].flatten(2).permute(0, 2, 1) # (bs, H*W, hidden_size)
            cnn_feature_list[i] = pos_embd_batch + cnn_feature_list[i] # position embedding
            cnn_feature_list[i] = self.layer_embd[i].unsqueeze(0).unsqueeze(0) + cnn_feature_list[i] # layer embedding

        output = self.transformer(cnn_feature_list, self.obj_query.unsqueeze(0).repeat((config["batch_size"])))
        return self.linear_class(output), self.linear_bbox(output).sigmoid()