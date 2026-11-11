# import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import A
class Conv_no(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size:int=3,
                 nolinear:bool=True
                 ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding= (kernel_size - 1) // 2, bias=False)
        self.nolinear = nolinear
        if self.nolinear:
            # self.act = nn.PReLU()
            self.act = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        if self.nolinear:
            x = self.act(x)
        return x

class DC(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam1 = nn.Parameter(torch.ones(1) * 0.5)
    def forward(self, out: torch.Tensor, kspace: torch.Tensor, mask: torch.Tensor):
        """
        out: B C H W
        kspace: B C H W
        mask: B 1 H W
        """
        out = out * (1 - mask) + kspace * mask * self.lam1 + out * mask * (1 - self.lam1)  
        return out


class istaBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 kernel_size:int=3,
                #  nolinear:bool=True
                 ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding= (kernel_size - 1) // 2, bias=False)

        self.convblock1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        )

        self.conv2 = nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1, bias=False)

        self.soft_thr = nn.Parameter(torch.ones(1) * 0.1)
        self.A = A(modality="2D")
        self.lambda_step = nn.Parameter(torch.ones(1) * 0.01)
  
    def forward(self, img, kspace0, mask, csm):

        ## I ignore a loss here if for paper adding the loss. (https://github.com/jianzhangcs/ISTA-Net-PyTorch)
        
        multi_img = self.A.expand_csm(img, csm)
        kspace_input = self.A.itok(multi_img)

        combine_kspace = kspace_input - self.lambda_step * (kspace_input - kspace0) * mask
        combine_img = self.A.combine_csm(self.A.ktoi(combine_kspace), csm)

        x_input = combine_img
        x = self.conv(x_input)
        x_forward = self.convblock1(x)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x_backward = self.convblock2(x)
        out = self.conv2(x_backward)

        x_pred = x_input + out

        return x_pred