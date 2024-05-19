import torch
import torch.nn as nn
from lib.models.AccumulatedToken.ATM_cs_p import ATM

x = torch.rand((1, 16, 2048))
model = ATM()
model(x)