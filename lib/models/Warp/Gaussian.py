import torch
import torch.nn as nn

class Gaussian_Fusing(nn.Module):
    def __init__(self, 
                 seqlen,
                 embed_dim,
                 mean=0.0, 
                 sigma=1.0):
        super().__init__()
        self.seqlen = seqlen
        self.mean = mean
        self.sigma = sigma

        self.shift_proj = nn.Linear(embed_dim, embed_dim)
        self.scale_proj = nn.Linear(embed_dim, embed_dim)

    def create_gaussian_kernel(self, kernel_size):
        x = torch.arange(kernel_size).float().cuda() - self.mean        # (x-mu)
        gauss_kernel = torch.exp(-x**2 / (2 * self.sigma**2))           # exp(-(x-mu)^2 / 2sig^2)
        gauss_kernel /= gauss_kernel.sum()  
        return gauss_kernel.flip(-1)
    
    def forward(self, x) :

        filtered_features = []
        for t in range(self.seqlen) :
            gauss_kernel = self.create_gaussian_kernel(t + 1)
            gauss_kernel = gauss_kernel.view(1, -1, 1)
            filtered_feature = torch.sum(x[:, :t+1] * gauss_kernel, dim=1)
            
            
            

        return 