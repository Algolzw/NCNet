import torch
import torch.nn as nn
import torch.nn.functional as F


class UpOnly(nn.Sequential):
    def __init__(self, scale):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError

        super(UpOnly, self).__init__(*m)


class NCNet(nn.Module):
    def __init__(self, n_feats=32, out_c=3, scale_factor=3):
        super(NCNet, self).__init__()

        ps_feat = out_c*(scale_factor**2)

        self.nearest_weight = torch.eye(out_c).repeat(1, scale_factor**2).reshape(ps_feat, out_c)
        self.nearest_weight = self.nearest_weight.unsqueeze(-1).unsqueeze(-1)
        
        # define body module
        self.body = nn.Sequential(
            nn.Conv2d(out_c, n_feats, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(n_feats, ps_feat, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(ps_feat, ps_feat, 3, 1, 1))
        
        self.upsample = UpOnly(scale_factor)

    def forward(self, x):
        x_res = F.conv2d(x, self.nearest_weight)
        x = self.body(x)
        x += x_res
        x = self.upsample(x)
        return x
