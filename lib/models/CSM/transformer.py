import torch
import torch.nn as nn
from functools import partial
from lib.models.trans_operator import Block

class ChannelTransformer(nn.Module):
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024,
            t_length=16, s_length=4, slice=4,
            h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0.):
        super().__init__()
        qkv_bias = True
        qk_scale = None
        self.slice = slice
        self.t_length = t_length
        self.s_length = s_length

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.pos_embed_t = nn.Parameter(torch.zeros(1, t_length, embed_dim))
        self.pos_embed_s = nn.Parameter(torch.zeros(1, 1, s_length, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim) 
    
    def channel_slice(self, x) :
        C = x.shape[-1]

        sliced_dim = C // self.slice
        sliced_tensor = torch.split(x, sliced_dim, dim=-1)  # l1, l2, l3, l4 : [B, T, D/4]
        sliced_tensor = torch.stack([tensor + self.pos_embed_t for tensor in sliced_tensor])    # [B, T, 4, D/4]
        sliced_tensor = sliced_tensor + self.pos_embed_s  
        
        sliced_tensor = torch.flatten(sliced_tensor, 1, 2)     # [B, 4T, 512]
        return sliced_tensor

    def forward(self, x):
        """
        x : [B, T, D]
        """
        B, T = x.shape[:2]
        x = self.channel_slice(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = x.reshape(B, T, -1)
        return x

class Transformer(nn.Module):
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024, \
            h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=16):
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

    def forward(self, x): 
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

class GlobalTransformer(nn.Module):
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024, \
            h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=27, \
            ):
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
                dim=embed_dim // 2, num_heads=h, mlp_hidden_dim=embed_dim * 2, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth // 2)])
        self.decoder_embed = nn.Linear(embed_dim, embed_dim // 2, bias=True)
        self.mask_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim // 2))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim // 2))
        self.decoder_norm = norm_layer(embed_dim // 2)

    def forward(self, x, is_train=True, mask_ratio=0.):
        if is_train:
            latent, mask, ids_restore = self.forward_encoder(x, mask_flag=True, mask_ratio=mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        else:
            latent, mask, ids_restore = self.forward_encoder(x, mask_flag=False,mask_ratio=0.)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        return pred, mask

    def forward_encoder(self, x, mask_flag=False, mask_ratio=0.):
        x = x + self.pos_embed
        if mask_flag:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
            # print('mask')
        else:
            mask = None
            ids_restore = None

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        if ids_restore is not None:
            mask_tokens = self.mask_tokens.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        else:
            x_ = x
        x = x_ + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device, dtype=torch.bool)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore).unsqueeze(-1) # assgin value from ids_restore
        
        return x_masked, mask, ids_restore