# Context Filtering Model
import torch
import torch.nn as nn

class CFM(nn.Module):
    def __init__(self, d_model) :
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2*d_model),
        )

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

    def forward(self, x_enc, context_feat) :
        """
        x_enc, context_feat     [B, T, C]

        return 
            fus_feat : [B, T, C]
        """
        B, T, C = x_enc.shape

        avg_x_enc = torch.mean(x_enc, dim=1, keepdim=True)
        avg_context_feat = torch.mean(context_feat, dim=1, keepdim=True)
        weights = self.encode(avg_x_enc + avg_context_feat).reshape(B, 1, C, 2)
        weights = weights.softmax(dim=-2)

        fus_feat = x_enc * weights[..., 0] + context_feat * weights[..., 1]

        return fus_feat

class Layer(nn.Module):
    def __init__(self, in_features, out_features) :
        super().__init__()
        self.proj = nn.Linear(in_features=in_features, out_features=out_features)
        self.norm = nn.LayerNorm(out_features)
        self.relu = nn.ReLU()

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

    def forward(self, x) :
        return self.relu(self.norm(self.proj(x)))

class U_CFM(nn.Module):
    def __init__(self, seqlen=16, sqe_seqlen=6, n_stage=2) :
        super().__init__()
        self.n_stage = n_stage
        interval = (seqlen-sqe_seqlen) // n_stage

        self.norm = nn.LayerNorm(2048)
        self.layers = nn.ModuleList()
        for n in range(n_stage):
            in_features = seqlen - n * interval
            out_features = seqlen - (n+1) * interval
            self.layers.append(Layer(in_features, out_features))

        for n in range(n_stage):
            in_features = sqe_seqlen + n * interval
            out_features = sqe_seqlen + (n+1) * interval
            self.layers.append(Layer(in_features, out_features))

    def forward(self, x_enc, context_feat):
        """
        x : [B, T, C]
        """
        x0 = x_enc + context_feat       # [B, C, 16]
        x0 = self.norm(x0)
        x0 = x0.permute(0, 2, 1)

        x1 = self.layers[0](x0)         # [B, C, 11]
        x2 = self.layers[1](x1)         # [B, C, 6]

        x3 = self.layers[2](x2) + x1   # [B, C, 11]
        x4 = self.layers[3](x3) + x0

        x4 = x4.permute(0, 2, 1)
        return x4