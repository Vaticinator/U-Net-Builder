import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision.utils import save_image

class UNet(nn.Module):

    
    def save_x(self, x, label="layer", wait=False):
        # save all features of the first image in batch
        save_image(x[0].view(-1,1,x.shape[2],x.shape[3]), "{}.png".format(label), nrow=int(x.shape[1]**0.5), normalize=True)
        if wait: input("Features saved. Press any key to continue.")

    def __init__(   self,
                    ch_in, ch_out, 
                    init_features = 64, 
                    u_blocks_amount = [1,2,2,6,2,2,1], 
                    u_blocks_variant = ['C','C','C','R','C','C','C'], 
                    u_blocks_resize = ['N','D','D','N','U','U','N'], 
                    u_connected = False,
                    use_dropout = False,
                    padding_mode = 'reflect',
                    fin_act = nn.Tanh()
        ):
        
        assert(len(u_blocks_amount) == len(u_blocks_variant) == len(u_blocks_resize))
        super(UNet, self).__init__()
        self.description = f'--------unet--------\nINIT Conv {ch_in} -> {init_features}\n'
        self.initial = nn.Sequential(
            nn.Conv2d(ch_in, init_features, kernel_size=7, padding=3, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(init_features),
            nn.GELU()
        )

        self.u_connected = u_connected
        if self.u_connected:
            u_blocks_resize_np = np.array(u_blocks_resize, dtype=str)
            self.connect_d = np.where(u_blocks_resize_np=='D')[0] #[1,2]
            self.connect_u = np.where(u_blocks_resize_np=='U')[0] #[4,5]
        
        features_in = init_features
        features_out = init_features
        self.network = nn.ModuleList([])
        for i in range(len(u_blocks_amount)):
            
            sampling = "up" if u_blocks_resize[i] == 'U' else "down" if u_blocks_resize[i] == 'D' else None
            features_out = features_out//2 if u_blocks_resize[i] == 'U' else features_out*2 if u_blocks_resize[i] == 'D' else features_out            
            self.description += f'{u_blocks_amount[i]}, {u_blocks_variant[i]}, {u_blocks_resize[i]}, {features_in} -> {features_out}\n'
            
            seq_list = []
            for n in range(u_blocks_amount[i]):
                
                if u_blocks_variant[i] == 'C':
                    seq_list.append(ConvBlock(features_in, features_out, sampling=sampling, use_dropout=use_dropout, padding_mode=padding_mode))
                elif u_blocks_variant[i] == 'R':
                    seq_list.append(ResidualBlock(features_in, features_out, sampling=sampling, use_dropout=use_dropout, padding_mode=padding_mode))
                elif u_blocks_variant[i] == 'E':
                    seq_list.append(SEResidualBlock(features_in, features_out, sampling=sampling, use_dropout=use_dropout, padding_mode=padding_mode))
                elif u_blocks_variant[i] == 'A':
                    seq_list.append(DualAttentionBlock(features_in))
                elif u_blocks_variant[i] == 'T':
                    seq_list.append(CrissCrossAttention(features_in))
                
                features_in = features_out
                sampling = None
            self.network.append(nn.Sequential(*seq_list))
        
        self.description += f'FINAL Conv {features_in} -> {ch_out}\nFINAL Act {fin_act}\n--------------------'
        self.final = nn.Sequential(
            nn.Conv2d(features_out, ch_out, kernel_size=7, padding=3, padding_mode=padding_mode),
            fin_act
        )
            
    def forward(self, x, debug=False):
        h = []
        x = self.initial(x)
        for i in range(len(self.network)):
            if self.u_connected and i in self.connect_d: h.append(x)
            x = self.network[i](x)
            if self.u_connected and i in self.connect_u: x = x + h.pop() * 0.1
            if debug: self.save_x(x, i)
        return self.final(x)

class ConvBlock(nn.Module):

    def __init__(self, ch_in, ch_out, sampling=None, use_dropout=False, padding_mode='reflect'):

        assert(sampling in ["down", None, "up"])
        super(ConvBlock, self).__init__()
        
        kernel_size = 5 if sampling == "up" else 3
        padding = 2 if sampling=="up" else 1
        stride = 2 if sampling=="down" else 1
        
        self.upscale = nn.Upsample(size=None, scale_factor=(2,2)) if sampling == "up" else nn.Identity()        
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, padding=padding, stride=stride, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.GELU()
        )

    def forward(self, x):
        x = self.upscale(x)
        return self.net(x)

class ResidualBlock(nn.Module):

    def __init__(self, ch_in, ch_out, sampling=None, use_dropout=False, padding_mode='reflect'):

        assert(sampling in ["down", None, "up"])
        super(ResidualBlock, self).__init__()
        
        kernel_size = 5 if sampling == "up" else 3
        padding = 2 if sampling=="up" else 1
        stride = 2 if sampling=="down" else 1

        self.upscale = nn.Upsample(size=None, scale_factor=(2,2)) if sampling == "up" else nn.Identity()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, padding=padding, stride=stride, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.GELU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(ch_out)
        )

        if sampling == "down":
            self.fit = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=2, padding_mode=padding_mode, bias=False),
                nn.BatchNorm2d(ch_out),
            )
        elif ch_in != ch_out:
            self.fit = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, padding_mode=padding_mode, bias=False),
                nn.BatchNorm2d(ch_out),
            )
        else:
            self.fit = nn.Identity()
        
        self.act = nn.GELU()

    def forward(self, x):
        x = self.upscale(x)
        out = self.fit(x) + self.net(x)       
        return self.act(out)
        
        
class SEResidualBlock(nn.Module):

    def __init__(self, ch_in, ch_out, sampling=None, use_dropout=False, padding_mode='reflect'):

        assert(sampling in ["down", None, "up"])
        super(SEResidualBlock, self).__init__()
        
        kernel_size = 5 if sampling == "up" else 3
        padding = 2 if sampling=="up" else 1
        stride = 2 if sampling=="down" else 1

        self.upscale = nn.Upsample(size=None, scale_factor=(2,2)) if sampling == "up" else nn.Identity()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, padding=padding, stride=stride, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.GELU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(ch_out)
        )

        if sampling == "down":
            self.fit = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=2, padding_mode=padding_mode, bias=False),
                nn.BatchNorm2d(ch_out),
            )
        elif ch_in != ch_out:
            self.fit = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, padding_mode=padding_mode, bias=False),
                nn.BatchNorm2d(ch_out),
            )
        else:
            self.fit = nn.Identity()
        
        self.squeeze_excitation = torchvision.ops.SqueezeExcitation(ch_in, ch_out)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.upscale(x)
        out = self.fit(x) + self.squeeze_excitation(self.net(x))
        return self.act(out)
        

# ---------------------------------------------------------------------------------------------------------------- Attention modules


class DualAttentionBlock(nn.Module):

    def __init__(self, ch):

        super(DualAttentionBlock, self).__init__()        
        self.pa = PositionAttention(ch)
        self.sdpa = ScaledDotProductAttention()

    def forward(self, x):    
        pa = self.pa(x)
        sdpa = self.sdpa(x)
        return sdpa + pa


class CrissCrossAttention(nn.Module):
    ''' Criss-Cross Attention Module '''
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def INF(self,B,H,W):
        return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width).to(x.get_device())).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x


class ScaledDotProductAttention(nn.Module):

    def __init__(self, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = attn_dropout

    def forward(self, x, mask=None):
        m_batchsize, d, height, width = x.size()
        q = x.view(m_batchsize, d, -1)
        k = x.view(m_batchsize, d, -1)
        v = x.view(m_batchsize, d, -1)

        attn = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout)
        output = attn.view(m_batchsize, d, height, width)

        return output
        

class PositionAttention(nn.Module):
    
    def __init__(self, in_dim, reduction_ratio=16):
        super(PositionAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim // reduction_ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // reduction_ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        attention = torch.bmm(proj_query, proj_key)
        attention = self.softmax(attention)        

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        out = torch.cat([avg_out, max_out], dim=1)
        out = self.fc(out)
        out = out + self.gamma * out.sigmoid()  # Apply sigmoid for scaling between 0 and 1
        return x * out
