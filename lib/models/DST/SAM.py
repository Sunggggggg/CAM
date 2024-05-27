"""Spatial information aggreate module"""
import torch
import torch.nn as nn

class SAM(nn.Module) :
    def __init__(self, d_model=256) :
        super().__init__()
        self.proj1 = nn.Linear(2048, d_model)
        self.proj2 = nn.Linear(d_model*3, d_model)
        self.act = nn.Tanh()
        self.proj3 = nn.Linear(d_model, d_model)
        self.proj4 = nn.Linear(d_model, 3)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x) :
        B = x.shape[0]

        x = self.proj1(x)   # [B, 3, D]
        _x = x
        x = x.flatten(-2)   # [B, 3D]
        x = self.proj2(x)   # [B, D]
        x = self.act(x)
        x = self.proj3(x)   # [B, D]
        x = self.act(x)
        x = self.proj4(x)   # [B, 3]
        x = self.act(x)
        x = self.softmax(x) # [B, 3]

        x = x.view(B, 1, 3)
        x = x @ _x

        return x