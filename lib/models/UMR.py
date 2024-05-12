import torch
import torch.nn as nn
from lib.models.transformer import Transformer

class Layer(nn.Module) :
    def __init__(self, 
                 in_seqlen, 
                 out_seqlen, 
                 in_dim, 
                 out_dim,
                 depth,
                 num_head=8,
                 drop_rate=0.1,
                 drop_path_rate=0.2,
                 attn_drop=0.,
                 stream="DOWN"
                 ) :
        super().__init__()
        self.stream = stream
        if stream == "DOWN":
            self.transformer = Transformer(depth=depth, embed_dim=in_dim, mlp_hidden_dim=in_dim*2,
                h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop, length=in_seqlen)
        else :
            self.transformer = Transformer(depth=depth, embed_dim=out_dim, mlp_hidden_dim=out_dim*2,
                h=num_head, drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop, length=out_seqlen)
        
        self.sampling = nn.Linear(in_seqlen, out_seqlen)
        
        self.c_proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
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
        """
        x : [B, T, C]
        """
        if self.stream == "DOWN":
            x = self.transformer(x)     # [B, T, C]
            x = x.permute(0, 2, 1)      # [B, C, T]
            x = self.sampling(x)        # [B, C, T/2]
            x = x.permute(0, 2, 1)      # [B, T/2, C]
            x = self.c_proj(x)          # [B, T/2, 2C]
            x = self.norm(x)            # [B, T/2, 2C]
            x = self.relu(x)            # [B, T/2, 2C]
        else :
            x = x.permute(0, 2, 1)      # [B, C, T]
            x = self.sampling(x)        # [B, C, 2T]
            x = x.permute(0, 2, 1)      # [B, 2T, C]
            x = self.c_proj(x)          # [B, 2T, C/2]
            x = self.norm(x)            # [B, 2T, C/2]
            x = self.relu(x)            # 
            x = self.transformer(x)     # [B, 2T, C/2]

        return x

"UNet Modeling for humanMeshRecostruction"

class UMR(nn.Module):
    def __init__(self,
                 seqlen=16,
                 d_model=256,
                 num_head=8,
                 drop_rate=0.1,
                 drop_path_rate=0.2,
                 attn_drop=0.
                 ):
        super().__init__()
        self.input_proj = nn.Linear(2048, d_model)
        self.output_proj = nn.Linear(d_model, 2048)

        self.down_proj = nn.ModuleList()
        self.encoder = nn.ModuleList()
        
        self.down1 = Layer(in_seqlen=seqlen, out_seqlen=seqlen//2, in_dim=d_model, out_dim=d_model*2, depth=1,
              num_head=num_head, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop=attn_drop, stream="DOWN")

        self.down2 = Layer(in_seqlen=seqlen//2, out_seqlen=seqlen//4, in_dim=d_model*2, out_dim=d_model*4, depth=2,
              num_head=num_head, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop=attn_drop, stream="DOWN")
        
        self.down3 = Layer(in_seqlen=seqlen//4, out_seqlen=seqlen//8, in_dim=d_model*4, out_dim=d_model*8, depth=3,
              num_head=num_head, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop=attn_drop, stream="DOWN")
        
        self.up3 = Layer(in_seqlen=seqlen//8, out_seqlen=seqlen//4, in_dim=d_model*8, out_dim=d_model*4, depth=3, 
              num_head=num_head, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop=attn_drop, stream="UP")
        
        self.up2 = Layer(in_seqlen=seqlen//4, out_seqlen=seqlen//2, in_dim=d_model*4, out_dim=d_model*2, depth=2,
              num_head=num_head, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop=attn_drop, stream="UP")
        
        self.up1 = Layer(in_seqlen=seqlen//2, out_seqlen=seqlen, in_dim=d_model*2, out_dim=d_model, depth=1,
              num_head=num_head, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop=attn_drop, stream="UP")

    def forward(self, x) :
        """
        x : [B, T, C]
        """
        x = self.input_proj(x)  # [B, T, 256]

        x1 = self.down1(x)      # [B, T, 256]
        x2 = self.down2(x1)     # [B, T/2, 512]
        x3 = self.down3(x2)     # [B, T/4, 1024]
        x4 = self.up3(x3)       # [B, T/4, 1024]
        x5 = self.up2(x4)       # [B, T/2, 512]
        x6 = self.up1(x5)       # [B, T, 256]

        x_out = self.output_proj(x6)

        return x_out

if __name__ == "__main__":
    x = torch.randn((1, 16, 2048))
    model = UMR()
    model(x)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)) 