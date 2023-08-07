import torch
from modeling.denim import *


if __name__ == "__main__":
    bs = 8
    resolution = 1024
    sz = 256
    k = 32 
    ps = 128  # 64
    ch = 9  # 15
    backbone_type = "style_wb" # "mixed_wb"

    I = torch.rand((bs, ch, resolution, resolution)).cuda()
    net = DeNIM_to_Canon(k, ch).cuda()
    net2 = DeNIM_wo_Fusion(k).cuda()
    E = AWBEncoder(
        sz=sz, k=k,
        backbone_type=backbone_type, backbone_inchnls=ch, backbone_ps=ps, 
        backbone_weights=f"weights/{backbone_type}_{ps}_{ch}.pth"
    ).cuda()
    d = E(I)
    print(d.shape)
    canon = net(I, d)
    print(canon.shape)
    out = net2(canon)
    print(out.shape)