import torch
from torch import nn
from kornia.geometry.transform import resize
from collections import OrderedDict

from modeling.backbone import WBnetIFRNet, WBnet


class DeNIM(nn.Module):
    def __init__(self, k) -> None:
        super().__init__()
        self.P = nn.Parameter(torch.empty((3, k)), requires_grad=True)
        self.Q = nn.Parameter(torch.empty((k, 3)), requires_grad=True)
        torch.nn.init.kaiming_normal_(self.P)
        torch.nn.init.kaiming_normal_(self.Q)
        self.k = k
        
    def forward(self, I, T):
        bs, _, H, W = I.shape
        x = torch.flatten(I, start_dim=2).transpose(1, 2)
        out = x @ self.P @ T.view(bs, self.k, self.k) @ self.Q
        out = out.view(bs, H, W, -1).permute(0, 3, 1, 2)
        return out
    
    
class DeNIM_to_Canon(nn.Module):
    def __init__(self, k, ch: int = 3) -> None:
        super().__init__()
        self.P = nn.Parameter(torch.empty((ch, k)), requires_grad=True)
        self.Q = nn.Parameter(torch.empty((k, ch)), requires_grad=True)
        self.R = nn.Parameter(torch.empty((ch, 3)), requires_grad=True)
        torch.nn.init.kaiming_normal_(self.P)
        torch.nn.init.kaiming_normal_(self.Q)
        torch.nn.init.kaiming_normal_(self.R)
        self.k = k
        
    def forward(self, I, T):
        bs, _, H, W = I.shape
        x = torch.flatten(I, start_dim=2).transpose(1, 2)
        out = x @ self.P @ T.view(bs, self.k, self.k) @ self.Q @ self.R
        out = out.view(bs, H, W, -1).permute(0, 3, 1, 2)
        return out
    

class DeNIM_wo_Fusion(nn.Module):
    def __init__(self, k, ch: int = 3) -> None:
        super().__init__()
        self.P = nn.Parameter(torch.empty((ch, k)), requires_grad=True)
        self.T = nn.Parameter(torch.empty((k, k)), requires_grad=True)
        self.Q = nn.Parameter(torch.empty((k, ch)), requires_grad=True)
        torch.nn.init.kaiming_normal_(self.P)
        torch.nn.init.kaiming_normal_(self.T)
        torch.nn.init.kaiming_normal_(self.Q)
        self.k = k
        
    def forward(self, I):
        bs, _, H, W = I.shape
        x = torch.flatten(I, start_dim=2).transpose(1, 2)
        out = x @ self.P @ self.T @ self.Q
        out = out.view(bs, H, W, -1).permute(0, 3, 1, 2)
        return out


class AWBEncoder(nn.Module):
    def __init__(
        self,
        sz,
        k,
        backbone_inchnls: int,
        backbone_ps: int,
        backbone_weights: str = None,
        backbone_type: str = "style_wb",
        device: str = "cuda"
    ) -> None:
        super().__init__()
        self.sz = sz
        assert backbone_type in ["mixed_wb", "style_wb"], f"Unrecognized backbone type: {backbone_type}"
        self.backbone_type = backbone_type
        self.backbone_inchnls = backbone_inchnls
        self.backbone_ps = backbone_ps
        self.backbone_weights = backbone_weights
        self.device = device
        self._1x1conv = torch.nn.Conv2d(64 if self.backbone_type == "mixed_wb" else 128, 1, 1)  # 128 for IFRNet, 64 for MixedWB
        self.act = torch.nn.GELU()
        
        self._init_backbone()
        self.D = nn.Linear(in_features=1024, out_features=k*k)

    def _init_backbone(self):
        self.backbone = WBnetIFRNet(inchnls=self.backbone_inchnls, ps=self.backbone_ps) \
            if self.backbone_type == "style_wb" else \
                WBnet(inchnls=self.backbone_inchnls)
        
        if self.backbone_weights is None:
            return
        state_dict = torch.load(self.backbone_weights, map_location=self.device)
        self.backbone.load_state_dict(state_dict)
    
    def forward(self, x):
        # x, shape: B x C x H x W
        with torch.no_grad():
            out = resize(x, (self.sz, self.sz), interpolation='bilinear')  # shape: B x C x sz x sz
            out = self.backbone(out)  # shape: B x 256 x 32 x 32 
        out = self.act(self._1x1conv(out))  # shape: B x 1 x 32 x 32
        out = torch.flatten(out, start_dim=1)  # shape: B x 1024
        return self.D(out)  # shape: B x k*k
