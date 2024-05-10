import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'


import torch
import importlib

def main(cfg):
    x = torch.randn((1, 16, 2048))
    model_module = importlib.import_module('.%s' % 'CAM', 'lib.models')
    model = model_module.CAM(
        seqlen=cfg.DATASET.SEQLEN, 
        d_model=cfg.MODEL.d_model, 
        num_head=cfg.MODEL.num_head, 
        spatial_n_layer=cfg.MODEL.spatial_n_layer,
    )

    return


if __name__ == "__main__" :
    main(None)
    