import torch
from einops import rearrange
import numpy as np



def ktoi(kspace):
    kspace = rearrange(kspace, 'b (coil two) h w -> b coil h w two', two=2).contiguous()
    kspace = torch.view_as_complex(kspace)
    kspace = torch.fft.ifftshift(kspace, dim=[2, 3])
    img = torch.fft.ifft2(kspace, dim=[2, 3])    
    img = torch.fft.fftshift(img, dim=[2, 3])
    img = torch.view_as_real(img)
    img = rearrange(img, 'b coil h w two -> b (coil two) h w')
    return img


def itok(img):
    img = rearrange(img, 'b (coil two) h w -> b coil h w two', two=2).contiguous()
    img = torch.view_as_complex(img)
    img = torch.fft.ifftshift(img, dim=[2, 3])
    kspace = torch.fft.fft2(img, dim=[2, 3])
    kspace = torch.fft.fftshift(kspace, dim=[2, 3])
    kspace = torch.view_as_real(kspace)
    kspace = rearrange(kspace, 'b coil h w two -> b (coil two) h w')
    return kspace


def rss(img):
    img = rearrange(img, 'b (coil two) h w -> b coil h w two', two=2).contiguous()
    img = torch.view_as_complex(img)
    img= torch.abs(img)
    img= torch.sqrt(torch.sum(torch.square(img), dim=1))
    return img  # B H W

def combine_csm(img, csm):
    # img: B C H W
    # csm: B C H W
    img = rearrange(img, 'b (coil two) h w -> b coil h w two', two=2).contiguous()
    csm = rearrange(csm, 'b (coil two) h w -> b coil h w two', two=2).contiguous()
    img = torch.view_as_complex(img)
    csm = torch.view_as_complex(csm)
    img = img * torch.conj(csm)
    img = torch.sum(img, dim=1, keepdim=True)
    img = torch.view_as_real(img)
    img = rearrange(img, 'b coil h w two  -> b (coil two) h w')
    return img
    
def expand_csm(img, csm):
    # img: B C H W
    # csm: B C H W
    img = rearrange(img, 'b (coil two) h w -> b coil h w two', two=2).contiguous()
    csm = rearrange(csm, 'b (coil two) h w -> b coil h w two', two=2).contiguous()
    img = torch.view_as_complex(img)
    csm = torch.view_as_complex(csm)
    img = img * csm
    img = torch.view_as_real(img)
    img = rearrange(img, 'b coil h w two -> b (coil two) h w')
    return img



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

