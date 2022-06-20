import torch.nn as nn
import torch
from PGKD_Net.lgam import LGAM
from PGKD_Net.cbam import CBAMBlock
from PGKD_Net.res2deform import CSE_Block,Up_Block_A,conv_block


class Unet_SE(nn.Module):
    def __init__(self, in_c, out_c, filter=64):
        super(Unet_SE, self).__init__()
        self.layer1 = conv_block(in_c, filter)  # 64
        self.layer2 = conv_block(filter, filter * 2)  # 128
        self.layer3 = conv_block(filter * 2, filter * 4)  # 256
        self.layer4 = conv_block(filter * 4, filter * 8)  # 512
        self.layer5 = conv_block(filter * 8, filter * 8)  # 512
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1)

        # first decoder
        self.up4 = Up_Block_A(512, 512, 1024, 512)
        self.up3 = Up_Block_A(512, 256, 512, 256)
        self.up2 = Up_Block_A(256, 128, 256, 128)
        self.up1 = Up_Block_A(128, 64, 128, 64)

        #self.se1 = LGAM(64, 32, 49)
        #self.se2 = LGAM(128, 16, 49)
        #self.se3 = LGAM(256, 8, 49)
        #self.se4 = LGAM(512, 4, 49)
        #self.se5 = LGAM(512, 2, 49)
        self.se1 = CSE_Block(64)
        self.se2 = CSE_Block(128)
        self.se3 = CSE_Block(256)
        self.se4 = CSE_Block(512)
        self.se5 = CSE_Block(512)
        self.last_activation = nn.Sigmoid()
        # second out conv
        # self.out_conv = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        #     nn.Conv2d(in_channels=32, out_channels=out_c, kernel_size=(1, 1)),
        #     nn.Sigmoid()
        # )
        self.out_conv = nn.Conv2d(64, out_c, kernel_size=1, stride=1, padding=0)
        # self.combine_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=out_c, kernel_size=1, padding=0),
        #     # nn.Sigmoid()
        # )

    def forward(self, x):
        # print("unetB input shape:", input_2.shape)
        en_x1 = self.layer1(x)  # 224,224,64
        en_x1 = self.se1(en_x1)  # 224,224,64
        # print("unetB en_x1 shape:", en_x1.shape)
        pool_x1 = self.pool(en_x1)
        # print("unetB pool_x1 shape:", pool_x1.shape)
        en_x2 = self.layer2(pool_x1)  # 112,112,128
        en_x2 = self.se2(en_x2)
        # print("unetB en_x2 shape:", en_x2.shape)
        pool_x2 = self.pool(en_x2)
        # print("unetB pool_x2 shape:", pool_x2.shape)
        en_x3 = self.layer3(pool_x2)  # 56,56,256
        en_x3 = self.se3(en_x3)
        # print("unetB en_x3 shape:", en_x3.shape)
        pool_x3 = self.pool(en_x3)
        # print("unetB pool_x3 shape:", pool_x3.shape)
        en_x4 = self.layer4(pool_x3)  # 28,28,512
        en_x4 = self.se4(en_x4)
        # print("unetB en_x4 shape:", en_x4.shape)
        pool_x4 = self.pool(en_x4)
        # print("unetB pool_x4 shape:", pool_x4.shape)
        en_x5 = self.layer5(pool_x4)  # 14,14, 512
        en_x5 = self.se5(en_x5)
        # print("unetB en_x5 shape:", en_x5.shape)
        # pool_x5 = self.pool(en_x5)
        # print("unetB pool_x5 shape:", pool_x5.shape)

        # aspp_out = self.aspp(en_x5)
        # print('up3', aspp_out.size(), pool_x4.size(), enc4.size())
        de_x4 = self.up4(en_x5, en_x4)
        # de_x4 = self.se4(de_x4)
        # print("unetB de_x4 shape:", de_x4.shape)

        de_x3 = self.up3(de_x4, en_x3)
        # de_x3 = self.se3(de_x3)
        # print("unetB de_x3 shape:", de_x3.shape)
        de_x2 = self.up2(de_x3, en_x2)
        # de_x2 = self.se2(de_x2)
        # print("unetB de_x2 shape:", de_x2.shape)
        de_x1 = self.up1(de_x2, en_x1)
        # de_x1 = self.se1(de_x1)
        # print("unetB de_x1 shape:", de_x1.shape)
        output2 = self.out_conv(de_x1)
        # output = torch.cat([output1, output2], dim=1)
        # output = self.combine_conv(output)
        output2 = self.last_activation(output2)
        return output2

class Unet_CBAM(nn.Module):
    def __init__(self, in_c, out_c, filter=64):
        super(Unet_CBAM, self).__init__()
        self.layer1 = conv_block(in_c, filter)  # 64
        self.layer2 = conv_block(filter, filter * 2)  # 128
        self.layer3 = conv_block(filter * 2, filter * 4)  # 256
        self.layer4 = conv_block(filter * 4, filter * 8)  # 512
        self.layer5 = conv_block(filter * 8, filter * 8)  # 512
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1)

        # first decoder
        self.up4 = Up_Block_A(512, 512, 1024, 512)
        self.up3 = Up_Block_A(512, 256, 512, 256)
        self.up2 = Up_Block_A(256, 128, 256, 128)
        self.up1 = Up_Block_A(128, 64, 128, 64)

        #self.se1 = LGAM(64, 32, 49)
        #self.se2 = LGAM(128, 16, 49)
        #self.se3 = LGAM(256, 8, 49)
        #self.se4 = LGAM(512, 4, 49)
        #self.se5 = LGAM(512, 2, 49)
        self.se1 = CBAMBlock(64)
        self.se2 = CBAMBlock(128)
        self.se3 = CBAMBlock(256)
        self.se4 = CBAMBlock(512)
        self.se5 = CBAMBlock(512)
        self.last_activation = nn.Sigmoid()
        # second out conv
        # self.out_conv = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        #     nn.Conv2d(in_channels=32, out_channels=out_c, kernel_size=(1, 1)),
        #     nn.Sigmoid()
        # )
        self.out_conv = nn.Conv2d(64, out_c, kernel_size=1, stride=1, padding=0)
        # self.combine_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=out_c, kernel_size=1, padding=0),
        #     # nn.Sigmoid()
        # )

    def forward(self, x):
        # print("unetB input shape:", input_2.shape)
        en_x1 = self.layer1(x)  # 224,224,64
        en_x1 = self.se1(en_x1)  # 224,224,64
        # print("unetB en_x1 shape:", en_x1.shape)
        pool_x1 = self.pool(en_x1)
        # print("unetB pool_x1 shape:", pool_x1.shape)
        en_x2 = self.layer2(pool_x1)  # 112,112,128
        en_x2 = self.se2(en_x2)
        # print("unetB en_x2 shape:", en_x2.shape)
        pool_x2 = self.pool(en_x2)
        # print("unetB pool_x2 shape:", pool_x2.shape)
        en_x3 = self.layer3(pool_x2)  # 56,56,256
        en_x3 = self.se3(en_x3)
        # print("unetB en_x3 shape:", en_x3.shape)
        pool_x3 = self.pool(en_x3)
        # print("unetB pool_x3 shape:", pool_x3.shape)
        en_x4 = self.layer4(pool_x3)  # 28,28,512
        en_x4 = self.se4(en_x4)
        # print("unetB en_x4 shape:", en_x4.shape)
        pool_x4 = self.pool(en_x4)
        # print("unetB pool_x4 shape:", pool_x4.shape)
        en_x5 = self.layer5(pool_x4)  # 14,14, 512
        en_x5 = self.se5(en_x5)
        # print("unetB en_x5 shape:", en_x5.shape)
        # pool_x5 = self.pool(en_x5)
        # print("unetB pool_x5 shape:", pool_x5.shape)

        # aspp_out = self.aspp(en_x5)
        # print('up3', aspp_out.size(), pool_x4.size(), enc4.size())
        de_x4 = self.up4(en_x5, en_x4)
        # de_x4 = self.se4(de_x4)
        # print("unetB de_x4 shape:", de_x4.shape)

        de_x3 = self.up3(de_x4, en_x3)
        # de_x3 = self.se3(de_x3)
        # print("unetB de_x3 shape:", de_x3.shape)
        de_x2 = self.up2(de_x3, en_x2)
        # de_x2 = self.se2(de_x2)
        # print("unetB de_x2 shape:", de_x2.shape)
        de_x1 = self.up1(de_x2, en_x1)
        # de_x1 = self.se1(de_x1)
        # print("unetB de_x1 shape:", de_x1.shape)
        output2 = self.out_conv(de_x1)
        # output = torch.cat([output1, output2], dim=1)
        # output = self.combine_conv(output)
        output2 = self.last_activation(output2)
        return output2

class Unet_LGAM(nn.Module):
    def __init__(self, in_c, out_c, filter=64):
        super(Unet_LGAM, self).__init__()
        self.layer1 = conv_block(in_c, filter)  # 64
        self.layer2 = conv_block(filter, filter * 2)  # 128
        self.layer3 = conv_block(filter * 2, filter * 4)  # 256
        self.layer4 = conv_block(filter * 4, filter * 8)  # 512
        self.layer5 = conv_block(filter * 8, filter * 8)  # 512
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1)

        # first decoder
        self.up4 = Up_Block_A(512, 512, 1024, 512)
        self.up3 = Up_Block_A(512, 256, 512, 256)
        self.up2 = Up_Block_A(256, 128, 256, 128)
        self.up1 = Up_Block_A(128, 64, 128, 64)

        self.se1 = CSE_Block(64)
        self.se2 = LGAM(128, 16, 49)
        self.se3 = LGAM(256, 8, 49)
        self.se4 = LGAM(512, 4, 49)
        self.se5 = LGAM(512, 2, 49)
        #self.se1 = CBAMBlock(64)
        #self.se2 = CBAMBlock(128)
        #self.se3 = CBAMBlock(256)
        #self.se4 = CBAMBlock(512)
        #self.se5 = CBAMBlock(512)
        self.last_activation = nn.Sigmoid()
        # second out conv
        # self.out_conv = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        #     nn.Conv2d(in_channels=32, out_channels=out_c, kernel_size=(1, 1)),
        #     nn.Sigmoid()
        # )
        self.out_conv = nn.Conv2d(64, out_c, kernel_size=1, stride=1, padding=0)
        # self.combine_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=out_c, kernel_size=1, padding=0),
        #     # nn.Sigmoid()
        # )

    def forward(self, x):
        # print("unetB input shape:", input_2.shape)
        en_x1 = self.layer1(x)  # 224,224,64
        en_x1 = self.se1(en_x1)  # 224,224,64
        # print("unetB en_x1 shape:", en_x1.shape)
        pool_x1 = self.pool(en_x1)
        # print("unetB pool_x1 shape:", pool_x1.shape)
        en_x2 = self.layer2(pool_x1)  # 112,112,128
        en_x2 = self.se2(en_x2)
        # print("unetB en_x2 shape:", en_x2.shape)
        pool_x2 = self.pool(en_x2)
        # print("unetB pool_x2 shape:", pool_x2.shape)
        en_x3 = self.layer3(pool_x2)  # 56,56,256
        en_x3 = self.se3(en_x3)
        # print("unetB en_x3 shape:", en_x3.shape)
        pool_x3 = self.pool(en_x3)
        # print("unetB pool_x3 shape:", pool_x3.shape)
        en_x4 = self.layer4(pool_x3)  # 28,28,512
        en_x4 = self.se4(en_x4)
        # print("unetB en_x4 shape:", en_x4.shape)
        pool_x4 = self.pool(en_x4)
        # print("unetB pool_x4 shape:", pool_x4.shape)
        en_x5 = self.layer5(pool_x4)  # 14,14, 512
        en_x5 = self.se5(en_x5)
        # print("unetB en_x5 shape:", en_x5.shape)
        # pool_x5 = self.pool(en_x5)
        # print("unetB pool_x5 shape:", pool_x5.shape)

        # aspp_out = self.aspp(en_x5)
        # print('up3', aspp_out.size(), pool_x4.size(), enc4.size())
        de_x4 = self.up4(en_x5, en_x4)
        # de_x4 = self.se4(de_x4)
        # print("unetB de_x4 shape:", de_x4.shape)

        de_x3 = self.up3(de_x4, en_x3)
        # de_x3 = self.se3(de_x3)
        # print("unetB de_x3 shape:", de_x3.shape)
        de_x2 = self.up2(de_x3, en_x2)
        # de_x2 = self.se2(de_x2)
        # print("unetB de_x2 shape:", de_x2.shape)
        de_x1 = self.up1(de_x2, en_x1)
        # de_x1 = self.se1(de_x1)
        # print("unetB de_x1 shape:", de_x1.shape)
        output2 = self.out_conv(de_x1)
        # output = torch.cat([output1, output2], dim=1)
        # output = self.combine_conv(output)
        output2 = self.last_activation(output2)
        return output2

class Unet(nn.Module):
    def __init__(self, in_c, out_c, filter=64):
        super(Unet, self).__init__()
        self.layer1 = conv_block(in_c, filter)  # 64
        self.layer2 = conv_block(filter, filter * 2)  # 128
        self.layer3 = conv_block(filter * 2, filter * 4)  # 256
        self.layer4 = conv_block(filter * 4, filter * 8)  # 512
        self.layer5 = conv_block(filter * 8, filter * 8)  # 512
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1)

        # first decoder
        self.up4 = Up_Block_A(512, 512, 1024, 512)
        self.up3 = Up_Block_A(512, 256, 512, 256)
        self.up2 = Up_Block_A(256, 128, 256, 128)
        self.up1 = Up_Block_A(128, 64, 128, 64)

        #self.se1 = LGAM(64, 32, 49)
        #self.se2 = LGAM(128, 16, 49)
        #self.se3 = LGAM(256, 8, 49)
        #self.se4 = LGAM(512, 4, 49)
        #self.se5 = LGAM(512, 2, 49)
        #self.se1 = CSE_Block(64)
        #self.se2 = CSE_Block(128)
        #self.se3 = CSE_Block(256)
        #self.se4 = CSE_Block(512)
        #self.se5 = CSE_Block(512)
        self.last_activation = nn.Sigmoid()
        # second out conv
        # self.out_conv = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        #     nn.Conv2d(in_channels=32, out_channels=out_c, kernel_size=(1, 1)),
        #     nn.Sigmoid()
        # )
        self.out_conv = nn.Conv2d(64, out_c, kernel_size=1, stride=1, padding=0)
        # self.combine_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=out_c, kernel_size=1, padding=0),
        #     # nn.Sigmoid()
        # )

    def forward(self, x):
        # print("unetB input shape:", input_2.shape)
        en_x1 = self.layer1(x)  # 224,224,64
        #en_x1 = self.se1(en_x1)  # 224,224,64
        # print("unetB en_x1 shape:", en_x1.shape)
        pool_x1 = self.pool(en_x1)
        # print("unetB pool_x1 shape:", pool_x1.shape)
        en_x2 = self.layer2(pool_x1)  # 112,112,128
        #en_x2 = self.se2(en_x2)
        # print("unetB en_x2 shape:", en_x2.shape)
        pool_x2 = self.pool(en_x2)
        # print("unetB pool_x2 shape:", pool_x2.shape)
        en_x3 = self.layer3(pool_x2)  # 56,56,256
        #en_x3 = self.se3(en_x3)
        # print("unetB en_x3 shape:", en_x3.shape)
        pool_x3 = self.pool(en_x3)
        # print("unetB pool_x3 shape:", pool_x3.shape)
        en_x4 = self.layer4(pool_x3)  # 28,28,512
        #en_x4 = self.se4(en_x4)
        # print("unetB en_x4 shape:", en_x4.shape)
        pool_x4 = self.pool(en_x4)
        # print("unetB pool_x4 shape:", pool_x4.shape)
        en_x5 = self.layer5(pool_x4)  # 14,14, 512
        #en_x5 = self.se5(en_x5)
        # print("unetB en_x5 shape:", en_x5.shape)
        # pool_x5 = self.pool(en_x5)
        # print("unetB pool_x5 shape:", pool_x5.shape)

        # aspp_out = self.aspp(en_x5)
        # print('up3', aspp_out.size(), pool_x4.size(), enc4.size())
        de_x4 = self.up4(en_x5, en_x4)
        # de_x4 = self.se4(de_x4)
        # print("unetB de_x4 shape:", de_x4.shape)

        de_x3 = self.up3(de_x4, en_x3)
        # de_x3 = self.se3(de_x3)
        # print("unetB de_x3 shape:", de_x3.shape)
        de_x2 = self.up2(de_x3, en_x2)
        # de_x2 = self.se2(de_x2)
        # print("unetB de_x2 shape:", de_x2.shape)
        de_x1 = self.up1(de_x2, en_x1)
        # de_x1 = self.se1(de_x1)
        # print("unetB de_x1 shape:", de_x1.shape)
        output2 = self.out_conv(de_x1)
        # output = torch.cat([output1, output2], dim=1)
        # output = self.combine_conv(output)
        output2 = self.last_activation(output2)
        return output2
