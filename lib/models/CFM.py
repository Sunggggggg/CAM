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
    def __init__(self, seqlen=16, sqe_seqlen=6 ,n_stage=2) :
        super().__init__()
        interval = (seqlen-sqe_seqlen) // n_stage

        layers = []
        for n in range(n_stage):
            in_features = seqlen - n * interval
            out_features = seqlen - (n+1) * interval
            layers.append(Layer(in_features, out_features))

        for n in range(n_stage):
            in_features = sqe_seqlen + n * interval
            out_features = sqe_seqlen + (n+1) * interval
            layers.append(Layer(in_features, out_features))
        
        self.u_net = nn.Sequential(*layers)

    def forward(self, x_enc, context_feat):
        """
        x : [B, T, C]
        """
        x = x_enc + context_feat

        x = x.permute(0, 2, 1)      # [B, C, T]
        x = self.u_net(x)
        x = x.permute(0, 2, 1) 

        return x