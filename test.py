import torch
import torch.nn as nn

x = nn.Parameter(torch.ones(1, 128) * 0.5, requires_grad=False)
print(x)