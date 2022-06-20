from re import S
import re
import torch
# from torch.autograd.grad_mode import F
import torch.nn.functional as F
import torch.nn as nn
# from dropblock import DropBlock2D
# from PGKD_Net.lgam import LGAM,LGAMV2
# from PGKD_Net.cbam import CBAMBlock
from PGKD_Net.res2deform import ASPP,Up_Block_A,conv_block,Bottle2neck,MSDMSA_bottle2neck,CSE_Block,Basic_Conv_Block

class Unet_A_trans(nn.Module):
    def __init__(self, in_c, base_block, trans_block, layers, img_size=128, baseWidth=28, scale=4, num_classes=1):
        super(Unet_A_trans, self).__init__()
        # first encoder
        # self.base_model = base_model
        self.inplanes = 32
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(in_c, 32, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(base_block, 32, layers[0])
        self.layer2 = self._make_layer(base_block, 64, layers[1])
        self.layer3 = self._make_layer(trans_block, 128, layers[2])
        self.layer4 = self._make_layer(trans_block, 256, layers[3], kernel_size=img_size // 8)
        self.layer5 = self._make_layer(trans_block, 256, layers[4], kernel_size=img_size // 16)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1)
        # self.se1 = CSE_Block(64)
        # self.se2 = CSE_Block(128)
        # self.se3 = CSE_Block(256)
        # self.se4 = CSE_Block(512)
        # self.se5 = CSE_Block(512)
        # self.se1 = LGAMV2(64, 32, 49)
        # self.se2 = LGAMV2(128, 16, 49)
        # self.se3 = LGAMV2(256, 8, 49)
        # self.se4 = LGAMV2(512, 4, 49)
        # self.se5 = LGAMV2(512, 2, 49)

        # first aspp
        self.aspp = ASPP(512, 512)

        # first decoder
        self.up4 = Up_Block_A(512, 512, 1024, 512)
        self.up3 = Up_Block_A(512, 256, 512, 256)
        self.up2 = Up_Block_A(256, 128, 256, 128)
        self.up1 = Up_Block_A(128, 64, 128, 64)

        # first out conv
        # self.out_conv = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        #     nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1)),
        #     nn.Sigmoid()
        # )
        # self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, blocks, kernel_size=1, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("unetA input shape:", input.shape)
        en_x1 = self.layer1(x)
        # en_x1 = self.se1(en_x1)
        pool_x1 = self.pool(en_x1)
        # print("unetA pool_x1 shape:", pool_x1.shape)
        en_x2 = self.layer2(pool_x1)
        # en_x2 = self.se2(en_x2)
        pool_x2 = self.pool(en_x2)
        # print("unetA pool_x2 shape:", pool_x2.shape)
        en_x3 = self.layer3(pool_x2)
        # en_x3 = self.se3(en_x3)
        pool_x3 = self.pool(en_x3)
        # print("unetA pool_x3 shape:", pool_x3.shape)
        en_x4 = self.layer4(pool_x3)
        # en_x4 = self.se4(en_x4)
        pool_x4 = self.pool(en_x4)
        # print("unetA pool_x4 shape:", pool_x4.shape)
        en_x5 = self.layer5(pool_x4)
        # en_x5 = self.se5(en_x5)
        # pool_x5 = self.pool(en_x5)
        # print("unetA pool_x5 shape:", pool_x5.shape)

        aspp_out = self.aspp(en_x5)
        # print("unetA aspp shape:", aspp_out.shape)

        de_x4 = self.up4(aspp_out, en_x4)
        # print("unetA de_x4 shape:", de_x4.shape)
        de_x3 = self.up3(de_x4, en_x3)
        # print("unetA de_x3 shape:", de_x3.shape)
        de_x2 = self.up2(de_x3, en_x2)
        # print("unetA de_x2 shape:", de_x2.shape)
        de_x1 = self.up1(de_x2, en_x1)

        
        # print("unetA de_x1 shape:", de_x1.shape)
        # ms_features = [en_x5, de_x4, de_x3, de_x2]
        encoder_features = [en_x1, en_x2, en_x3, en_x4]
        decoder_features = [de_x1, de_x2, de_x3, de_x4]
        # output = self.out_conv(de_x1)
        # output = F.sigmoid(output)
        # print("unetA output shape:", output.shape)

        return input, de_x1, encoder_features, decoder_features


class Unet_A_res(nn.Module):
    def __init__(self, in_c, base_block, layers, baseWidth=28, scale=4, num_classes=1):
        super(Unet_A_res, self).__init__()
        # first encoder
        # self.base_model = base_model
        self.inplanes = 32
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(in_c, 32, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(base_block, 32, layers[0])
        self.layer2 = self._make_layer(base_block, 64, layers[1])
        self.layer3 = self._make_layer(base_block, 128, layers[2])
        self.layer4 = self._make_layer(base_block, 256, layers[3])
        self.layer5 = self._make_layer(base_block, 256, layers[4])
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1)
        # self.se1 = CSE_Block(64)
        # self.se2 = CSE_Block(128)
        # self.se3 = CSE_Block(256)
        # self.se4 = CSE_Block(512)
        # self.se5 = CSE_Block(512)
        # self.se1 = LGAMV2(64, 32, 49)
        # self.se2 = LGAMV2(128, 16, 49)
        # self.se3 = LGAMV2(256, 8, 49)
        # self.se4 = LGAMV2(512, 4, 49)
        # self.se5 = LGAMV2(512, 2, 49)

        # first aspp
        self.aspp = ASPP(512, 512)

        # first decoder
        self.up4 = Up_Block_A(512, 512, 1024, 512)
        self.up3 = Up_Block_A(512, 256, 512, 256)
        self.up2 = Up_Block_A(256, 128, 256, 128)
        self.up1 = Up_Block_A(128, 64, 128, 64)

        # first out conv
        # self.out_conv = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        #     nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1)),
        #     nn.Sigmoid()
        # )
        # self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, blocks, kernel_size=1, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("unetA input shape:", input.shape)
        en_x1 = self.layer1(x)
        # en_x1 = self.se1(en_x1)
        pool_x1 = self.pool(en_x1)
        # print("unetA pool_x1 shape:", pool_x1.shape)
        en_x2 = self.layer2(pool_x1)
        # en_x2 = self.se2(en_x2)
        pool_x2 = self.pool(en_x2)
        # print("unetA pool_x2 shape:", pool_x2.shape)
        en_x3 = self.layer3(pool_x2)
        # en_x3 = self.se3(en_x3)
        pool_x3 = self.pool(en_x3)
        # print("unetA pool_x3 shape:", pool_x3.shape)
        en_x4 = self.layer4(pool_x3)
        # en_x4 = self.se4(en_x4)
        pool_x4 = self.pool(en_x4)
        # print("unetA pool_x4 shape:", pool_x4.shape)
        en_x5 = self.layer5(pool_x4)
        # en_x5 = self.se5(en_x5)
        # pool_x5 = self.pool(en_x5)
        # print("unetA pool_x5 shape:", pool_x5.shape)

        aspp_out = self.aspp(en_x5)
        # print("unetA aspp shape:", aspp_out.shape)

        de_x4 = self.up4(aspp_out, en_x4)
        # print("unetA de_x4 shape:", de_x4.shape)
        de_x3 = self.up3(de_x4, en_x3)
        # print("unetA de_x3 shape:", de_x3.shape)
        de_x2 = self.up2(de_x3, en_x2)
        # print("unetA de_x2 shape:", de_x2.shape)
        de_x1 = self.up1(de_x2, en_x1)

        
        # print("unetA de_x1 shape:", de_x1.shape)
        # ms_features = [en_x5, de_x4, de_x3, de_x2]
        encoder_features = [en_x1, en_x2, en_x3, en_x4]
        decoder_features = [de_x1, de_x2, de_x3, de_x4]
        # output = self.out_conv(de_x1)
        # output = F.sigmoid(output)
        # print("unetA output shape:", output.shape)

        return input, de_x1, encoder_features, decoder_features

class Unet_A(nn.Module):
    def __init__(self, in_c, out_c, filter=64):
        super(Unet_A, self).__init__()
        self.layer1 = conv_block(in_c, filter)  # 64
        self.layer2 = conv_block(filter, filter * 2)  # 128
        self.layer3 = conv_block(filter * 2, filter * 4)  # 256
        self.layer4 = conv_block(filter * 4, filter * 8)  # 512
        self.layer5 = conv_block(filter * 8, filter * 8)  # 512
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1)

        self.aspp = ASPP(512, 512)

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
        #self.se1 = LGAMV2(64, 32, 49)
        # self.se2 = LGAMV2(128, 16, 49)
        # self.se3 = LGAMV2(256, 8, 49)
        # self.se4 = LGAMV2(512, 4, 49)
        # self.se5 = LGAMV2(512, 2, 49)
        self.se1 = CSE_Block(64)
        self.se2 = CSE_Block(128)
        self.se3 = CSE_Block(256)
        self.se4 = CSE_Block(512)
        self.se5 = CSE_Block(512)

        # self.out_conv = nn.Conv2d(64, out_c, kernel_size=1, stride=1, padding=0)
        # self.combine_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=out_c, kernel_size=1, padding=0),
        #     # nn.Sigmoid()
        # )

    def forward(self, x):
        # print("unetB input shape:", input_2.shape)
        input = x
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

        aspp_out = self.aspp(en_x5)
        # print('up3', aspp_out.size(), pool_x4.size(), enc4.size())
        de_x4 = self.up4(aspp_out, en_x4)
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
        encoder_features = [en_x1, en_x2, en_x3, en_x4]
        decoder_features = [de_x1, de_x2, de_x3, de_x4]
        # output = self.out_conv(de_x1)
        # output = torch.cat([output1, output2], dim=1)
        # output = self.combine_conv(output)

        return input, de_x1, encoder_features, decoder_features

class Unet_B(nn.Module):
    def __init__(self, in_c, out_c, filter=64):
        super(Unet_B, self).__init__()
        self.layer1 = conv_block(in_c, filter)  # 64
        self.layer2 = conv_block(filter, filter * 2)  # 128
        self.layer3 = conv_block(filter * 2, filter * 4)  # 256
        self.layer4 = conv_block(filter * 4, filter * 8)  # 512
        self.layer5 = conv_block(filter * 8, filter * 8)  # 512
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1)

        self.aspp = ASPP(512, 512)

        # first decoder
        self.up4 = Up_Block_A(512, 512, 1024, 512) #1024+256=1280
        self.up3 = Up_Block_A(512, 256, 512, 256) # 512+128= 640
        self.up2 = Up_Block_A(256, 128, 256, 128)
        self.up1 = Up_Block_A(128, 64, 128, 64) # 128+32

        #self.se1 = LGAM(64, 32, 49)
        #self.se2 = LGAM(128, 16, 49)
        #self.se3 = LGAM(256, 8, 49)
        #self.se4 = LGAM(512, 4, 49)
        #self.se5 = LGAM(512, 2, 49)
        # self.se1 = CSE_Block(64)
        # self.se2 = CSE_Block(128)
        # self.se3 = CSE_Block(256)
        # self.se4 = CSE_Block(512)
        # self.se5 = CSE_Block(512)

        self.csff1 = FG_CSFF(64)
        self.csff2 = FG_CSFF(128)
        self.csff3 = FG_CSFF(256)
        self.csff4 = FG_CSFF(512)
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

    def forward(self, input, output1, encoder_features, decoder_features):
        enc1, enc2, enc3, enc4 = encoder_features
        dec1, dec2, dec3, dec4 = decoder_features
        # input_2 = input * output1
        # print("unetB input shape:", input_2.shape)
        en_x1 = self.layer1(output1)  # 224,224,64
        # en_x1 = self.csff1(enc1, dec1, en_x1)
        # en_x1 = self.se1(en_x1)  # 224,224,64
        # print("unetB en_x1 shape:", en_x1.shape)
        pool_x1 = self.pool(en_x1)
        # print("unetB pool_x1 shape:", pool_x1.shape)
        en_x2 = self.layer2(pool_x1)  # 112,112,128
        # en_x2 = self.csff2(enc2, dec2, en_x2)
        # en_x2 = self.se2(en_x2)
        # print("unetB en_x2 shape:", en_x2.shape)
        pool_x2 = self.pool(en_x2)
        # print("unetB pool_x2 shape:", pool_x2.shape)
        en_x3 = self.layer3(pool_x2)  # 56,56,256
        # en_x3 = self.csff3(enc3, dec3, en_x3)
        # en_x3 = self.se3(en_x3)
        # print("unetB en_x3 shape:", en_x3.shape)
        pool_x3 = self.pool(en_x3)
        # print("unetB pool_x3 shape:", pool_x3.shape)
        en_x4 = self.layer4(pool_x3)  # 28,28,512
        # en_x4 = self.csff4(enc4, dec4, en_x4)
        # en_x4 = self.se4(en_x4)
        # print("unetB en_x4 shape:", en_x4.shape)
        pool_x4 = self.pool(en_x4)
        # print("unetB pool_x4 shape:", pool_x4.shape)
        en_x5 = self.layer5(pool_x4)  # 14,14, 512
        # en_x5 = self.se5(en_x5)
        # print("unetB en_x5 shape:", en_x5.shape)
        # pool_x5 = self.pool(en_x5)
        # print("unetB pool_x5 shape:", pool_x5.shape)

        aspp_out = self.aspp(en_x5)
        # print('up3', aspp_out.size(), pool_x4.size(), enc4.size())
        de_x4 = self.up4(aspp_out, self.csff4(enc4, dec4, en_x4))
        # de_x4 = self.se4(de_x4)
        # print("unetB de_x4 shape:", de_x4.shape)

        de_x3 = self.up3(de_x4, self.csff3(enc3, dec3, en_x3))
        # de_x3 = self.se3(de_x3)
        # print("unetB de_x3 shape:", de_x3.shape)
        de_x2 = self.up2(de_x3, self.csff2(enc2, dec2, en_x2))
        # de_x2 = self.se2(de_x2)
        # print("unetB de_x2 shape:", de_x2.shape)
        de_x1 = self.up1(de_x2, self.csff1(enc1, dec1, en_x1))
        # de_x1 = self.se1(de_x1)
        # print("unetB de_x1 shape:", de_x1.shape)
        output2 = self.out_conv(de_x1)
        # output = torch.cat([output1, output2], dim=1)
        # output = self.combine_conv(output)

        return output2

class Unet_B_wo_csff(nn.Module):
    def __init__(self, in_c, out_c, filter=64):
        super(Unet_B_wo_csff, self).__init__()
        self.layer1 = conv_block(in_c, filter)  # 64
        self.layer2 = conv_block(filter, filter * 2)  # 128
        self.layer3 = conv_block(filter * 2, filter * 4)  # 256
        self.layer4 = conv_block(filter * 4, filter * 8)  # 512
        self.layer5 = conv_block(filter * 8, filter * 8)  # 512
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1)

        self.aspp = ASPP(512, 512)

        # first decoder
        self.up4 = Up_Block_A(512, 512, 1024, 512) #1024+256=1280
        self.up3 = Up_Block_A(512, 256, 512, 256) # 512+128= 640
        self.up2 = Up_Block_A(256, 128, 256, 128)
        self.up1 = Up_Block_A(128, 64, 128, 64) # 128+32

        # self.csff1 = FG_CSFF(64)
        # self.csff2 = FG_CSFF(128)
        # self.csff3 = FG_CSFF(256)
        # self.csff4 = FG_CSFF(512)
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

    def forward(self, input, output1):
        # enc1, enc2, enc3, enc4 = encoder_features
        # dec1, dec2, dec3, dec4 = decoder_features
        # input_2 = input * output1
        # print("unetB input shape:", input_2.shape)
        en_x1 = self.layer1(output1)  # 224,224,64
        # en_x1 = self.csff1(enc1, dec1, en_x1)
        # en_x1 = self.se1(en_x1)  # 224,224,64
        # print("unetB en_x1 shape:", en_x1.shape)
        pool_x1 = self.pool(en_x1)
        # print("unetB pool_x1 shape:", pool_x1.shape)
        en_x2 = self.layer2(pool_x1)  # 112,112,128
        # en_x2 = self.csff2(enc2, dec2, en_x2)
        # en_x2 = self.se2(en_x2)
        # print("unetB en_x2 shape:", en_x2.shape)
        pool_x2 = self.pool(en_x2)
        # print("unetB pool_x2 shape:", pool_x2.shape)
        en_x3 = self.layer3(pool_x2)  # 56,56,256
        # en_x3 = self.csff3(enc3, dec3, en_x3)
        # en_x3 = self.se3(en_x3)
        # print("unetB en_x3 shape:", en_x3.shape)
        pool_x3 = self.pool(en_x3)
        # print("unetB pool_x3 shape:", pool_x3.shape)
        en_x4 = self.layer4(pool_x3)  # 28,28,512
        # en_x4 = self.csff4(enc4, dec4, en_x4)
        # en_x4 = self.se4(en_x4)
        # print("unetB en_x4 shape:", en_x4.shape)
        pool_x4 = self.pool(en_x4)
        # print("unetB pool_x4 shape:", pool_x4.shape)
        en_x5 = self.layer5(pool_x4)  # 14,14, 512
        # en_x5 = self.se5(en_x5)
        # print("unetB en_x5 shape:", en_x5.shape)
        # pool_x5 = self.pool(en_x5)
        # print("unetB pool_x5 shape:", pool_x5.shape)

        aspp_out = self.aspp(en_x5)
        # print('up3', aspp_out.size(), pool_x4.size(), enc4.size())
        de_x4 = self.up4(aspp_out, en_x4)
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

        return output2

class Unet_B_trans(nn.Module):
    def __init__(self, in_c, out_c, base_block, trans_block, layers, img_size=128, baseWidth=28, scale=4):
        super(Unet_B_trans, self).__init__()
        self.inplanes = 32
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(in_c, 32, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(base_block, 32, layers[0])
        self.layer2 = self._make_layer(base_block, 64, layers[1])
        self.layer3 = self._make_layer(trans_block, 128, layers[2])
        self.layer4 = self._make_layer(trans_block, 256, layers[3], kernel_size=img_size // 8)
        self.layer5 = self._make_layer(trans_block, 256, layers[4], kernel_size=img_size // 16)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1)

        self.aspp = ASPP(512, 512)

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
        # self.csff_enc1 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        # self.csff_enc2 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        # self.csff_enc3 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        # self.csff_enc4 = nn.Conv2d(512, 512, kernel_size=1, bias=False) 
        # self.csff_dec1 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        # self.csff_dec2 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        # self.csff_dec3 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        # self.csff_dec4 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.csff1 = CSFF(64)
        self.csff2 = CSFF(128)
        self.csff3 = CSFF(256)
        self.csff4 = CSFF(512)
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
    def _make_layer(self, block, planes, blocks, kernel_size=1, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def forward(self, feature, encoder_features, decoder_features):
        enc1, enc2, enc3, enc4 = encoder_features
        dec1, dec2, dec3, dec4 = decoder_features
        # input_2 = input * output1
        # x = self.conv1(input_2)
        # x = self.bn1(x)
        # x = self.relu(x)
        # print("unetB input shape:", input_2.shape)
        en_x1 = self.layer1(feature)#+self.csff1(enc1, dec1)  # 224,224,64
        en_x1 = self.se1(en_x1)  # 224,224,64
        # print("unetB en_x1 shape:", en_x1.shape)
        pool_x1 = self.pool(en_x1)
        # print("unetB pool_x1 shape:", pool_x1.shape)
        en_x2 = self.layer2(pool_x1)#+self.csff2(enc2, dec2)  # 112,112,128
        en_x2 = self.se2(en_x2)
        # print("unetB en_x2 shape:", en_x2.shape)
        pool_x2 = self.pool(en_x2)
        # print("unetB pool_x2 shape:", pool_x2.shape)
        en_x3 = self.layer3(pool_x2)#+self.csff3(enc3, dec3)  # 56,56,256
        en_x3 = self.se3(en_x3)
        # print("unetB en_x3 shape:", en_x3.shape)
        pool_x3 = self.pool(en_x3)
        # print("unetB pool_x3 shape:", pool_x3.shape)
        en_x4 = self.layer4(pool_x3)#+self.csff4(enc4, dec4)
        # print("unetB en_x4 shape:", en_x4.shape)
        pool_x4 = self.pool(en_x4)
        # print("unetB pool_x4 shape:", pool_x4.shape)
        en_x5 = self.layer5(pool_x4)  # 14,14, 512
        en_x5 = self.se5(en_x5)
        # print("unetB en_x5 shape:", en_x5.shape)
        # pool_x5 = self.pool(en_x5)
        # print("unetB pool_x5 shape:", pool_x5.shape)

        aspp_out = self.aspp(en_x5)
        # print('up3', aspp_out.size(), pool_x4.size(), enc4.size())
        de_x4 = self.up4(aspp_out, self.csff4(enc4, dec4, en_x4))
        # de_x4 = self.se4(de_x4)
        # print("unetB de_x4 shape:", de_x4.shape)

        de_x3 = self.up3(de_x4, self.csff3(enc3, dec3, en_x3))
        # de_x3 = self.se3(de_x3)
        # print("unetB de_x3 shape:", de_x3.shape)
        de_x2 = self.up2(de_x3, self.csff2(enc2, dec2, en_x2))
        # de_x2 = self.se2(de_x2)
        # print("unetB de_x2 shape:", de_x2.shape)
        de_x1 = self.up1(de_x2, self.csff1(enc1, dec1, en_x1))
        # de_x1 = self.se1(de_x1)
        # print("unetB de_x1 shape:", de_x1.shape)
        output2 = self.out_conv(de_x1)
        # output = torch.cat([output1, output2], dim=1)
        # output = self.combine_conv(output)

        return output2

class ResDeTransDoubleUnet_lgam(nn.Module):
    def __init__(self, in_c, out_c, img_size=224):
        super(ResDeTransDoubleUnet_lgam, self).__init__()
        # self.base_model = base_model
        self.unet1 = Unet_A_trans(Bottle2neck, MSDMSA_bottle2neck, [2, 2, 2, 2, 2], img_size)
        self.unet2 = Unet_B(in_c, out_c, 64)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        input, output1, encoder_features = self.unet1(x)
        input2 = torch.sigmoid(output1)
        output2 = self.unet2(input, input2, encoder_features)
        # output1 = self.last_activation(output1)
        output2 = self.last_activation(output2)
        return input2, output2

class ResDeTransDoubleUnetV2_lgam(nn.Module):
    def __init__(self, in_c, out_c, img_size=224):
        super(ResDeTransDoubleUnetV2_lgam, self).__init__()
        # self.base_model = base_model
        self.unet1 = Unet_A(in_c, out_c, 64)
        self.unet2 = Unet_B_trans(in_c, out_c, Bottle2neck, MSDMSA_bottle2neck, [2, 2, 2, 2, 2], img_size)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        input, output1, encoder_features = self.unet1(x)
        input2 = torch.sigmoid(output1)
        output2 = self.unet2(input, input2, encoder_features)
        # output1 = self.last_activation(output1)
        output2 = self.last_activation(output2)
        return input2, output2

class SPG(nn.Module):
    def __init__(self, in_c, out_c, class_num):
        super(SPG, self).__init__()
        self.out_conv = nn.Conv2d(in_c, class_num, 1)
        self.conv2 = nn.Conv2d(class_num, in_c, 1)
        self.conv3 = Basic_Conv_Block(in_c, in_c, kernel_size=1, padding=0)
        self.conv4 = Basic_Conv_Block(in_c, out_c, kernel_size=1, padding=0)
        self.activation = nn.Sigmoid()

    def forward(self,x):
        x1 = self.out_conv(x)
        attn = self.activation(self.conv2(x1))
        transform = self.conv3(x)
        # print(attn.size(), transform.size(), flush=True)
        attn = attn*transform
        out = self.conv4(attn+x)
        return x1, out

class MS_SPG(nn.Module):
    def __init__(self, in_c, out_c, class_num):
        super(MS_SPG, self).__init__()
        filters = [64, 128, 256, 512, 512] 
        self.CatChannels=filters[0]
        in_c = self.CatChannels*len(filters)
        self.out_conv = nn.Conv2d(in_c, class_num, 1)
        self.conv2 = nn.Conv2d(class_num, in_c, 1)
        self.conv3 = Basic_Conv_Block(in_c, in_c, kernel_size=1, padding=0)
        self.conv4 = Basic_Conv_Block(in_c, out_c, kernel_size=1, padding=0)

        
        self.en5_de1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.en5_de1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1, bias=False)
        self.en5_de1_bn = nn.BatchNorm2d(self.CatChannels)

        self.de4_de1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.de4_de1_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1, bias=False)
        self.de4_de1_bn = nn.BatchNorm2d(self.CatChannels)

        self.de3_de1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.de3_de1_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1, bias=False)
        self.de3_de1_bn = nn.BatchNorm2d(self.CatChannels)

        self.de2_de1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.de2_de1_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1, bias=False)
        self.de2_de1_bn = nn.BatchNorm2d(self.CatChannels)

        self.activation = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, dec_lists, x):
        enc5, dec4, dec3, dec2 = dec_lists
        enc5_dec1 = self.relu(self.en5_de1_bn(self.en5_de1_conv(self.en5_de1(enc5))))
        dec4_dec1 = self.relu(self.de4_de1_bn(self.de4_de1_conv(self.de4_de1(dec4))))
        dec3_dec1 = self.relu(self.de3_de1_bn(self.de3_de1_conv(self.de3_de1(dec3))))
        dec2_dec1 = self.relu(self.de2_de1_bn(self.de2_de1_conv(self.de2_de1(dec2))))
        feature = torch.cat([x, dec2_dec1, dec3_dec1, dec4_dec1, enc5_dec1], dim=1)

        x1 = self.out_conv(feature)
        attn = self.activation(self.conv2(x1))
        transform = self.conv3(feature)
        # print(attn.size(), transform.size(), flush=True)
        attn = attn*transform
        out = self.conv4(attn+feature)
        return self.activation(x1), out


class CSFF(nn.Module):
    def __init__(self, in_c):
        super(CSFF, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_c)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_c)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(in_c, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.conv1 = nn.Conv2d(in_c, in_c, 1, bias=True)
        self.conv2 = nn.Conv2d(in_c, in_c, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, enc, dec, shotcut):
        fusion = self.relu(self.conv1(enc)+self.conv2(dec))
        g1 = self.W_g(fusion)
        x1 = self.W_x(shotcut)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        out = shotcut*psi
        return out

class Sk_Block(nn.Module):
    def __init__(self, in_c, reduce_ratio=4, L=32):
        super().__init__()
        self.d=max(L, in_c//reduce_ratio)
        self.refine = res_block(in_c, in_c)
        # self.conv1 = nn.Conv2d(in_c, in_c, 1, bias=True)
        # self.conv2 = nn.Conv2d(in_c, in_c, 1, bias=True)
        # self.conv1 = Basic_Conv_Block(in_c, in_c, 1, 0, 1)
        # self.conv2 = Basic_Conv_Block(in_c, in_c, 1, 0, 1)
        self.conv3 = nn.Conv2d(in_c, in_c // reduce_ratio, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc=nn.Linear(in_c, self.d)
        self.fcs=nn.ModuleList([])
        for i in range(2):
            self.fcs.append(
                nn.Sequential(
                    nn.Linear(self.d, in_c),
                    nn.Dropout(0.3),
                )
                
            )
        self.softmax=nn.Softmax(dim=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, enc, dec):
        # x1 = self.conv1(enc)
        # x2 = self.conv2(dec)
        feats=torch.stack([enc,dec],0)
        #fuse
        fusion = enc+dec
        bs, c, _, _ = fusion.size()
        S = fusion.mean(-1).mean(-1)
        Z=self.fc(S)
        #attn
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs,c,1,1))
        attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1
        #fuse
        V=(attention_weughts*feats).sum(0)

        return V

class SAPblock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=2*in_c, out_channels=in_c,dilation=1,kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=in_c, out_channels=in_c//2,dilation=1,kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_c//2, out_channels=2,dilation=1,kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, enc, dec):
        fusion = torch.cat([enc, dec], dim=1)
        attn = self.relu(self.conv1(fusion))
        attn = self.relu(self.conv2(attn))
        attn = self.conv3(attn)
        # print('attn:',attn.size())
        # attn = F.softmax(attn, dim=1)

        att_1=attn[:,0,:,:].unsqueeze(1)
        att_2=attn[:,1,:,:].unsqueeze(1)

        out = att_1*enc+att_2*dec
        return out

class res_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(res_block,self).__init__()
        self.res = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=1,stride=1,padding=0,bias=False),
                nn.BatchNorm2d(ch_out))
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        # self.main = conv_block(ch_in,ch_out)
        # self.bn2 = nn.BatchNorm2d(ch_out)

    def forward(self,x):
        res_x = self.res(x)
        main_x = self.relu(self.bn1(self.conv1(x)))
        main_x = self.bn2(self.conv2(main_x))
        out = res_x.add(main_x)
        out = self.relu(out)
        return out

class reverse_fusion(nn.Module):
    def __init__(self, in_c) -> None:
        super().__init__()
        self.dp_conv1 = nn.Conv2d(in_c, in_c, kernel_size=1,stride=1,padding=0,groups=in_c,bias=False)
        self.relu = nn.ReLU(inplace=True)
        # self.sk_block = Sk_Block(in_c)

    def forward(self, enc, dec):
        ba = self.relu(self.dp_conv1(dec))
        ba = (-1*torch.sigmoid(ba))+1
        attn_enc = enc.mul(ba)
        out = dec+attn_enc
        # fusion = self.sk_block(attn_enc, dec)
        return out

class Feature_selection(nn.Module):
    def __init__(self,in_c, out_c, reduce_ratio=4) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.reduce_ratio = reduce_ratio
        self.conv2 = nn.Conv2d(out_c, out_c // self.reduce_ratio, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(out_c // self.reduce_ratio, out_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1 = Basic_Conv_Block(in_c, out_c, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature):
        reduce_f = self.conv1(feature)
        attn = self.conv3(self.relu(self.conv2(self.avg_pool(reduce_f))))
        attn = torch.sigmoid(attn)
        out = reduce_f*attn + reduce_f
        return out

class FG_CSFF(nn.Module):
    def __init__(self, in_c, reduce_ratio=4):
        super(FG_CSFF, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.reduce_ratio = reduce_ratio
        # self.conv3 = nn.Conv2d(in_c, in_c // self.reduce_ratio, kernel_size=1, stride=1, padding=0, bias=True)
        # self.conv4 = nn.Conv2d(in_c // self.reduce_ratio, in_c, kernel_size=1, stride=1, padding=0, bias=True)
        self.sk_block = Sk_Block(in_c)
        self.sk_block2 = Sk_Block(in_c)
        # self.ra = reverse_fusion(in_c)
        # self.sk_block = SAPblock(in_c)
        # self.W_g = nn.Sequential(
        #     nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(in_c)
        # )
        # self.W_x = nn.Sequential(
        #     nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(in_c)
        # )
        # self.psi = nn.Sequential(
        #     nn.Conv2d(in_c, 1, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(1),
        #     nn.Sigmoid()
        # )
        # self.fs = Feature_selection(in_c, in_c//2)
        # self.refine = res_block(in_c, in_c)
        # self.conv1 = nn.Conv2d(in_c, in_c, 1, bias=True)
        # self.conv2 = nn.Conv2d(in_c, in_c, 1, bias=True)
        # self.gamma = nn.Parameter(torch.zeros(1))
        #self.relu = nn.ReLU(inplace=True)

    def forward(self, enc, dec, shortcut):
        # fusion = self.conv1(enc)+self.conv2(dec)
        # fusion = self.sk_block(enc, dec)
        fusion = self.sk_block(enc, dec)
        fusion2 = self.sk_block(fusion, shortcut)
        #cse
        # se = torch.sigmoid(self.conv4(self.relu(self.conv3(self.avg_pool(fusion)))))
        # se = se*shotcut
        # g1 = self.W_g(fusion)
        # x1 = self.W_x(shotcut)
        # psi = self.relu(g1+x1)
        # psi = self.psi(psi)
        # out = shotcut*psi
        # out = self.refine(out)
        # out = torch.cat([fusion, shortcut], dim=1)
        # out = fusion + shotcut
        # out = enx + self.gamma*fusion
        return fusion2

class ResDeTransDoubleUnetV2_spg(nn.Module):
    def __init__(self, in_c, out_c, img_size=224):
        super(ResDeTransDoubleUnetV2_spg, self).__init__()
        # self.base_model = base_model
        self.unet1 = Unet_A(in_c, out_c, 64)
        self.spg = SPG(64, 32, out_c)
        self.unet2 = Unet_B_trans(in_c, out_c, Bottle2neck, MSDMSA_bottle2neck, [2, 2, 2, 2, 2], img_size)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        input, output1, encoder_features, decoder_features = self.unet1(x)
        output1, feature = self.spg(output1)
        #input2 = torch.sigmoid(output1)
        output2 = self.unet2(feature, encoder_features, decoder_features)
        # output1 = self.last_activation(output1)
        output2 = self.last_activation(output2)
        return output1, output2

class ResDeTransDoubleUnet_spg(nn.Module):
    def __init__(self, in_c, out_c, img_size=224):
        super(ResDeTransDoubleUnet_spg, self).__init__()
        # self.base_model = base_model
        self.unet1 = Unet_A_trans(in_c, Bottle2neck, MSDMSA_bottle2neck, [2, 2, 2, 2, 2], img_size)
        self.spg = SPG(64, 32, out_c)
        self.unet2 = Unet_B(32, out_c, 64)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        input, output1, encoder_features, decoder_features = self.unet1(x)
        output1, feature = self.spg(output1)
        # input2 = torch.sigmoid(output1)
        output2 = self.unet2(input, feature, encoder_features, decoder_features)
        # output1 = self.last_activation(output1)
        # output2 = self.last_activation(output2)
        return output1, output2

class ResDeTransDoubleUnet_spg_wo_csff(nn.Module):
    def __init__(self, in_c, out_c, img_size=224):
        super(ResDeTransDoubleUnet_spg_wo_csff, self).__init__()
        # self.base_model = base_model
        self.unet1 = Unet_A_trans(in_c, Bottle2neck, MSDMSA_bottle2neck, [2, 2, 2, 2, 2], img_size)
        self.spg = SPG(64, 32, out_c)
        self.unet2 = Unet_B_wo_csff(32, out_c, 64)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        input, output1, encoder_features, decoder_features = self.unet1(x)
        output1, feature = self.spg(output1)
        # input2 = torch.sigmoid(output1)
        output2 = self.unet2(input, feature)
        # output1 = self.last_activation(output1)
        # output2 = self.last_activation(output2)
        return output1, output2

class ResDoubleUnet(nn.Module):
    def __init__(self, in_c, out_c, img_size=224):
        super(ResDoubleUnet, self).__init__()
        # self.base_model = base_model
        self.unet1 = Unet_A_res(in_c, Bottle2neck, [2, 2, 2, 2, 2])
        self.spg = SPG(64, 32, out_c)
        self.unet2 = Unet_B_wo_csff(32, out_c, 64)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        input, output1, encoder_features, decoder_features = self.unet1(x)
        output1, feature = self.spg(output1)
        # input2 = torch.sigmoid(output1)
        output2 = self.unet2(input, feature)
        # output1 = self.last_activation(output1)
        # output2 = self.last_activation(output2)
        return output1, output2

class ResDeTransDoubleUnet_spg_wo_trans(nn.Module):
    def __init__(self, in_c, out_c, img_size=224):
        super(ResDeTransDoubleUnet_spg_wo_trans, self).__init__()
        # self.base_model = base_model
        self.unet1 = Unet_A_res(in_c, Bottle2neck, [2, 2, 2, 2, 2])
        self.spg = SPG(64, 32, out_c)
        self.unet2 = Unet_B(32, out_c, 64)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        input, output1, encoder_features, decoder_features = self.unet1(x)
        output1, feature = self.spg(output1)
        # input2 = torch.sigmoid(output1)
        output2 = self.unet2(input, feature, encoder_features, decoder_features)
        # output1 = self.last_activation(output1)
        # output2 = self.last_activation(output2)
        return output1, output2