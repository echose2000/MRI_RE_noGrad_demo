import torch
import torch.nn as nn 
from .basic import Conv_no, DC, istaBlock
from utils import A

class Basicmodel(nn.Module):
    def __init__(self, 
                coils:int=15, 
                hidden_channels:int=32,
                num_layers:int=3,
                kernel_size:int=3,
                iter_num:int=2,
                share:bool = True
                ):
        super().__init__()

        assert num_layers >= 2
        two = 2
        self.A = A(modality="2D")

        in_channels = [coils * two ,*[hidden_channels] * (num_layers - 1)]
        out_channels = [*[hidden_channels] * (num_layers - 1), coils * two]
        self.model = nn.ModuleList()
        self.dc = nn.ModuleList()
        model = nn.Sequential(*[Conv_no(in_channel,out_channel, nolinear=False if i == num_layers - 1 else True, kernel_size=kernel_size)        
                                for i, (in_channel, out_channel) in enumerate(zip(in_channels, out_channels))])
        
        dc = DC()
      
       
        for i in range(iter_num):
            if not share:
                model = nn.Sequential(*[Conv_no(in_channel,out_channel, nolinear=False if i == num_layers - 1 else True, kernel_size=kernel_size)        
                                for i, (in_channel, out_channel) in enumerate(zip(in_channels, out_channels))])
        
                dc = DC()
            
            self.model.append(model)
            self.dc.append(dc)

    def forward(self, kspace:torch.Tensor, mask:torch.Tensor, csm:torch.Tensor, kspace0:torch.Tensor) -> torch.Tensor:
        out = kspace
        
        for dc, model in zip(self.dc, self.model):
            out = self.A.ktoi(out)
            #TODO Don't forget to change channels !!!
            # out = self.A.combine_csm(out, csm) 
            out = model(out)
            # out = self.A.expand_csm(out, csm)
            out = self.A.itok(out)
            out = dc(out, kspace0, mask)
        return out
    


class ISTAPlus(nn.Module):
    def __init__(self, 
                coils:int=15, 
                hidden_channels:int=32,
                num_layers:int=3,
                kernel_size:int=3,
                iter_num:int=2,
                share:bool = True
                ):
        super().__init__()

        self.A = A(modality="2D")
    
       
        self.model = nn.ModuleList()

        model = istaBlock(in_channels=2, mid_channels=hidden_channels, kernel_size=kernel_size)
        for i in range(iter_num):
            if not share:
                model = istaBlock(in_channels=2, mid_channels=hidden_channels, kernel_size=kernel_size)
            self.model.append(model)
        
    def forward(self, kspace:torch.Tensor, mask:torch.Tensor, csm:torch.Tensor) -> torch.Tensor:
        out = kspace

        img = self.A.combine_csm(self.A.ktoi(out), csm)
        for model in self.model:
            # out = self.A.ktoi(out)
            img = model(img, kspace, mask, csm)
            # out = self.A.itok(out)
        multi_img = self.A.expand_csm(img, csm)
        out = self.A.itok(multi_img)
        return out
