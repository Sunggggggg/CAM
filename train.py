import torch

from lib.models.CAM import CAM

if __name__ == "__main__" :
    x = torch.randn((1, 16, 2048))
    model = CAM()
    y = model(x)
    
    print(y.shape)