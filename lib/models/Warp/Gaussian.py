import torch
import torch.nn as nn

class Gaussian_Fusing(nn.Module):
    def __init__(self, 
                 seqlen,
                 embed_dim,
                 mean=0.0, 
                 sigma=2.0):
        super().__init__()
        self.seqlen = seqlen
        self.mean = mean
        self.sigma = sigma

        self.affine_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), 
            nn.ReLU()
        )
        self.shift_proj = nn.Linear(embed_dim, embed_dim)
        self.scale_proj = nn.Linear(embed_dim, embed_dim)

    def create_gaussian_kernel(self, kernel_size):
        x = torch.arange(kernel_size).float().cuda() - self.mean        # (x-mu)
        gauss_kernel = torch.exp(-x**2 / (2 * self.sigma**2))           # exp(-(x-mu)^2 / 2sig^2)
        gauss_kernel /= gauss_kernel.sum()  
        return gauss_kernel.flip(-1)
    
    def forward_direction(self, x) :
        warpped_features = []
        for t in range(self.seqlen) :
            gauss_kernel = self.create_gaussian_kernel(t + 1)
            gauss_kernel = gauss_kernel.view(1, -1, 1)
            filtered_feature = torch.sum(x[:, :t+1] * gauss_kernel, dim=1)  # [B, D]

            shift = self.shift_proj(filtered_feature)
            scale = self.scale_proj(filtered_feature)

            affin_warp = self.affine_proj(x[:, t])
            warpped_features.append(scale * affin_warp + shift)
        warpped_features = torch.stack(warpped_features, dim=1)

        return warpped_features
    
    def backward_direction(self, x):
        x = x.flip(dim=1)

        warpped_features = []
        for t in range(self.seqlen) :
            gauss_kernel = self.create_gaussian_kernel(t + 1)
            gauss_kernel = gauss_kernel.view(1, -1, 1)
            filtered_feature = torch.sum(x[:, :t+1] * gauss_kernel, dim=1)  # [B, D]

            shift = self.shift_proj(filtered_feature)
            scale = self.scale_proj(filtered_feature)

            warpped_features.append(scale * x[:, t] + shift)
        return warpped_features

    def forward(self, x) :
        projed_feat = self.forward_direction(x)

        return projed_feat