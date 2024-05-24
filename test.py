import torch
import torch.nn as nn
from lib.models.CSM import CSM

x = torch.rand((1, 16, 2048))
model = CSM()
model(x)