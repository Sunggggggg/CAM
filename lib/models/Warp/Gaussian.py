import torch
import torch.nn as nn

class Gaussian_Fusing(nn.Module):
    def __init__(self, 
                 seqlen=16,
                 mean=0.0, 
                 sigma=1.0):
        super().__init__()
        self.seqlen = seqlen
        self.mean = mean
        self.sigma = sigma

    def create_gaussian_kernel(self, kernel_size):
        x = torch.arange(kernel_size).float().cuda() - self.mean        # (x-mu)
        gauss_kernel = torch.exp(-x**2 / (2 * self.sigma**2))           # exp(-(x-mu)^2 / 2sig^2)
        gauss_kernel /= gauss_kernel.sum()  
        return gauss_kernel.flip(-1)
    
    def forward(self, x) :
        for t in range(self.seqlen) :
            self.create_gaussian_kernel(t)

        return 