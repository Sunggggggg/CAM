# Context Accmulation Model
import torch
import torch.nn as nn

class CAM(nn.Module) :
    def __init__(self,
                 seqlen=16,
                 d_model=2048,
                 d_token=128,
                 alpha=0.9,
                 ) :
        super().__init__()
        self.seqlen = seqlen
        assert d_model % seqlen == 0, f"{d_model} % {seqlen} = 0" 

        self.alpha = nn.Parameter(torch.ones(1, d_token) * alpha, requires_grad=False)

        self.proj_enc = nn.Linear(d_model, d_token)
        self.proj_dec = nn.Linear(d_token, d_model)
        self.norm_enc = nn.LayerNorm(d_token)
        self.norm_dec = nn.LayerNorm(d_model)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x_enc) :
        """
        x_enc : [B, T, D]

        return :
            context_feat : [B, T, D]
        """
        split_x_enc = self.proj_enc(x_enc)         # [B, T, d]
        split_x_enc = self.norm_enc(split_x_enc)

        context_tokens = []
        for t in range(self.seqlen) :
            if t == self.seqlen - 1 :
                context_token = split_x_enc[:, t]   # [B, d]
            else :
                context_token = (split_x_enc[:, t] + self.alpha * split_x_enc[:, t+1]) / (1 + self.alpha)
            context_tokens.append(context_token)
        context_feat = torch.stack(context_tokens, dim=1)  # [B, T, d]
        context_feat = self.proj_dec(context_feat)         # [B, T, D]
        context_feat = self.norm_dec(context_feat)         


        return context_feat