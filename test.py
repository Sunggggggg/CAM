import torch
import torch.nn as nn
from lib.models.OnlyConv.Conv import OC

x = torch.rand((1, 16, 2048))
model = OC()
model(x)