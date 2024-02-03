import torch
from torch import nn

config = {
    "batch_size": 64
}



class DETR(nn.Module):
    '''
    Compared to the pseudocode in DETR paper, 
    code here take batch size into consideration.
    '''
    def __init__(self, n_class, hidden_size, n_head, 
                n_encoder, n_decoder, backbone, N):
        super().__init__()
        # self.n_class = n_class
        # self.hidden_size = hidden_size
        # self.n_head = n_head
        # self.n_encoder = n_encoder
        # self.n_encoder = n_decoder
        # self.N = N

        # Assume that the backbone output feature map 
        # that transformer block need
        self.backbone = backbone
        self.linear_class = nn.Linear(hidden_size, n_class+1) # background class
        self.linear_bbox = nn.Linear(hidden_size, 4) # (center_x, center_y, height, width)
        # batch_first=True, takes (batch_size, seq_len, embd_size) as input
        # otherwise takes (seq_len, batch_size, embd_size)
        self.transformer = nn.Transformer(hidden_size, n_head, n_encoder, n_decoder, batch_first=True)

        self.obj_query = nn.Parameter(torch.rand(N, hidden_size))
        # make sure K >= max(*featureMap.shape[-2:]) (i.e. H & W)
        self.row_embd = nn.Parameter(torch.rand(K, hidden_size//2))
        self.col_embd = nn.Parameter(torch.rand(K, hidden_size//2))

    def forward(self, input):
        '''
        Take fixed size picture as input, 
        output class, tuple of bbox.
        bbox = [0,1]^4, which means the scale on the original image.
        '''
        cnn_feature = self.backbone(input)
        # by our assume, channel == hidden_size
        # otherwise a 1*1 conv may be used
        bs, channel, H, W = cnn_feature.shape 

        # position embedding
        row_repeat = self.row_embd[:H].unsqueeze(1).repeat(1, W, 1) # (H, W, hidden_size//2)
        col_repeat = self.col_embd[:W].unsqueeze(0).repeat(H, 1, 1) # (H, W, hidden_size//2)
        pos_embd = torch.cat([row_repeat, col_repeat], dim=-1).flatten(0, 1) # (H*W, hidden_size)
        pos_embd_batch = pos_embd.unsqueeze(0).repeat((config["batch_size"])) # (bs, H*W, hidden_size)

        cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1) # (bs, H*W, hidden_size)
        output = self.transformer(pos + cnn_feature, self.obj_query.unsqueeze(0).repeat((config["batch_size"])))
        return self.linear_class(output), self.linear_bbox(output).sigmoid()