import torch
from lib.data_utils._img_utils import get_single_image

def load_batch_image(img_name):
    """
    img_name : [B, T, ]
    """
    img_batch = []
    B, T = img_name.shape
    for b in range(B):
        for t in range(T):
            img = get_single_image(img_name[b, t])
            img_batch.append(img)
    img_batch = torch.from_numpy(img).reshape(B, T, *img.shape)
    return img_batch

