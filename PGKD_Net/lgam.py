import torch
import torch.nn as nn
from math import log

class Global_attention(nn.Module):
    def __init__(self, inplanes, reduce_ratio=4):
        super(Global_attention, self).__init__()
        self.iplanes = inplanes
        self.reduce_ratio = reduce_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(inplanes, inplanes // self.reduce_ratio, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(inplanes // self.reduce_ratio, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        out = input * x
        return out

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        # print('x:', x.size(), flush=True)
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        # print('y', y.size(), flush=True)
        y=self.conv(y) #bs,1,c
        # print('y2', y.size(), flush=True)
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        
        return x*y.expand_as(x)


class LGAM(nn.Module):
    def __init__(self, inplanes, patch_size, nums, kernel_size=5, gamma=2, b=1):
        super(LGAM, self).__init__()
        self.patch_size = patch_size
        self.inplanes = inplanes
        self.nums = nums
        self.gamma = gamma
        self.b = b
        self.k = int(abs((log(self.inplanes, 2) + self.b) / self.gamma))
        self.unfolder = nn.Unfold(kernel_size=patch_size, dilation=1, padding=0, stride=patch_size)
        # self.unfolder = nn.Fold(output_size=)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        # self.reduce_ratio = reduce_ratio
        self.conv1 = nn.Conv1d(self.nums, self.nums, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(self.nums, self.nums, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1,
                               padding=(kernel_size - 1) // 2)
        self.conv4 = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1,
                               padding=(kernel_size - 1) // 2)
        # self.conv2 = nn.Conv1d(inplanes * self.nums // self.reduce_ratio, inplanes * self.nums, kernel_size=1, stride=1,
        #                        padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.global_attn = ECAAttention(kernel_size=kernel_size)

    def forward(self, x):
        B, C, H, W = x.size()
        # num = (H / self.patch_size) * (W / self.patch_size)
        # split
        patches = self.unfolder(x)  # b,nums,featrues
        patches = patches.transpose(-1, -2).reshape(B, C * self.nums, self.patch_size, self.patch_size)  # adaptive
        # local attention
        avg_response = self.avg_pool(patches).reshape(B, self.nums, self.inplanes)
        max_response = self.max_pool(patches).reshape(B, self.nums, self.inplanes)
        avg_out = self.relu(self.conv1(avg_response))
        max_out = self.relu(self.conv2(max_response))
        # patch_response = self.conv2(patch_response)
        patch_response = torch.sigmoid(max_out + avg_out).reshape(B, self.nums * self.inplanes, 1, 1)
        patches = patch_response * patches
        # fold
        patches = patches.reshape(B, self.nums, C * self.patch_size * self.patch_size).transpose(-1, -2)

        # local patch attention
        #avg_patch = torch.mean(patches, dim=-2, keepdim=True)  # b, 1, num
        #max_patch, _ = torch.max(patches, dim=-2, keepdim=True)  # b, 1, num
        #avg_patch = self.relu(self.conv3(avg_patch))  # b,1, num
        #max_patch = self.relu(self.conv4(max_patch))
        #patch_response = torch.sigmoid(avg_patch + max_patch)  # b,1,num
        #patches = patch_response * patches  # b,c,num

        out = nn.Fold(output_size=(H, W), kernel_size=self.patch_size, stride=self.patch_size)(patches)  # B,C,H,W

        # global
        out = self.global_attn(out)
        return out

class PatchAttentionModule(nn.Module):
    
    def __init__(self,patch_size, nums, d_model=512, kernel_size=3):
        super().__init__()
        self.patch_size = patch_size
        self.inplanes = d_model
        self.nums = nums
        self.cnn = nn.Conv2d(d_model,d_model,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.unfolder = nn.Unfold(kernel_size=patch_size, dilation=1, padding=0, stride=patch_size)
        self.proj = nn.Conv2d(self.inplanes, self.nums, kernel_size=patch_size, stride=patch_size)  #embeding
        self.scale = self.inplanes**(-0.5)
        self.dropout = nn.Dropout(0.1)
        #self.pa=ScaledDotProductAttention(d_model,d_k=d_model,d_v=d_model,h=1)
    
    def forward(self, x):
        bs,c,h,w = x.size()
        y=self.cnn(x)
        patches = self.unfolder(y).transpose(1, 2)  # b, num, c
        attn = self.proj(y).flatten(2).transpose(1,2)
        attn = self.scale * attn
        attn = attn.softmax(-1) # b,num,num
        attn = self.dropout(attn)

        out = torch.matmul(attn, patches).transpose(1, 2)
        out = nn.Fold(output_size=(h, w), kernel_size=self.patch_size, stride=self.patch_size)(out)
        # y=y.view(bs,c,-1).permute(0,2,1) #bs,h*w,c
        # y=self.pa(y,y,y) #bs,h*w,c
        return out

class PatchChannelAttentionModule(nn.Module):
    
    def __init__(self, patch_size, nums, d_model=512, kernel_size=3):
        super().__init__()
        self.patch_size = patch_size
        self.inplanes = d_model
        self.nums = nums
        self.cnn = nn.Conv2d(d_model,d_model,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.unfolder = nn.Unfold(kernel_size=patch_size, dilation=1, padding=0, stride=patch_size)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.conv1 = nn.Conv1d(self.nums, self.nums, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(self.nums, self.nums, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.relu = nn.ReLU(inplace=True)
        #self.pa=ScaledDotProductAttention(d_model,d_k=d_model,d_v=d_model,h=1)
    
    def forward(self, x):
        bs,c,h,w = x.size()
        y=self.cnn(x)
        patches = self.unfolder(y).transpose(1, 2)  # b, num, c
        patches = patches.reshape(bs, c * self.nums, self.patch_size, self.patch_size)  # adaptive
        # local attention
        avg_response = self.avg_pool(patches).reshape(bs, self.nums, self.inplanes)
        max_response = self.max_pool(patches).reshape(bs, self.nums, self.inplanes)
        avg_out = self.relu(self.conv1(avg_response))
        max_out = self.relu(self.conv2(max_response))
        # patch_response = self.conv2(patch_response)
        patch_response = torch.sigmoid(max_out + avg_out).reshape(bs, self.nums * self.inplanes, 1, 1)
        patches = patch_response * patches
        # fold
        patches = patches.reshape(bs, self.nums, c * self.patch_size * self.patch_size).transpose(-1, -2)

        out = nn.Fold(output_size=(h, w), kernel_size=self.patch_size, stride=self.patch_size)(patches)
        # y=y.view(bs,c,-1).permute(0,2,1) #bs,h*w,c
        # y=self.pa(y,y,y) #bs,h*w,c
        return out

class LGAMV2(nn.Module):
    def __init__(self, inplanes, patch_size, nums, kernel_size=5, gamma=2, b=1):
        super(LGAMV2, self).__init__()
        self.patch_size = patch_size
        self.inplanes = inplanes
        self.nums = nums
        self.gamma = gamma
        self.b = b
        self.k = int(abs((log(self.inplanes, 2) + self.b) / self.gamma))
        self.pa = PatchAttentionModule(self.patch_size, self.nums, self.inplanes)
        self.pca = PatchChannelAttentionModule(self.patch_size, self.nums, self.inplanes, kernel_size)
        # self.unfolder = nn.Fold(output_size=)
        # self.reduce_ratio = reduce_ratio
        # self.conv2 = nn.Conv1d(inplanes * self.nums // self.reduce_ratio, inplanes * self.nums, kernel_size=1, stride=1,
        #                        padding=0, bias=False)
        self.global_attn = ECAAttention(kernel_size=kernel_size)

    def forward(self, x):
        B, C, H, W = x.size()
        out1 = self.pca(x)
        out2 = self.pa(x)
        # global
        out = self.global_attn((out1+out2))
        return out