import torch
import torch.nn as nn
from functools import partial

from lib.models.trans_operator import Block

class ED_Transformer(nn.Module):
    def __init__(self, depth=3, embed_dim=128, mlp_hidden_dim=256, h=8, 
                 drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=16):
        super().__init__()
        qkv_bias = True
        qk_scale = None

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)]) 
        self.norm = norm_layer(embed_dim)

        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=h, mlp_hidden_dim=embed_dim * 2, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=True)
        #self.decoder_pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))
        self.decoder_norm = norm_layer(embed_dim)
    
    def forward_encoder(self, x) :
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x) :
        x = self.decoder_embed(x)
        x = x + self.pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        return x

    def forward(self, x):
        """
        Input 
            x : [B, T, D]
        Return
            x : [B, T, D]
        """
        x = self.forward_encoder(x)
        x = self.forward_decoder(x)
        
        return x