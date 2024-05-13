import torch
from lib.models.UMR import UMR

x = torch.rand((1, 16, 2048))
model = UMR()
model(x)