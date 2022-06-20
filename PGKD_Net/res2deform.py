import torch
import torch.nn as nn
import math
import torch.nn.functional as F
# from dropblock import DropBlock2D
from PGKD_Net.Detrans.DeformableTrans import DeformableTransformer
from PGKD_Net.Detrans.position_encoding import build_position_encoding
from PGKD_Net.spconv import SPConv_3x3
# from PGKD_Net.lgam import LGAM

class conv_block(nn.Module):
    '''
        aggregation of conv operation
        conv-bn-relu-conv-bn-relu
        Example:
            input:(B,C,H,W)
            conv_block(C,out)
            conv_block(input)
            rerturn (B,out,H,W)
    '''

    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            # DropBlock2D(block_size=3, drop_prob=0.3),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            # DropBlock2D(block_size=3, drop_prob=0.3),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.se = CSE_Block(ch_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        return x


class Basic_Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
        super(Basic_Conv_Block, self).__init__()
        self.normal_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False,
                      dilation=dilation),
            #DropBlock2D(block_size=3, drop_prob=0.3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.normal_conv(x)


class Up_Block_A(nn.Module):
    def __init__(self, in_channels, up_channels, concat_channels, out_channels, upsample_method="transpose", up=True):
        """
        :param in_channels: 指的是输入的通道数
        :param up_channels: 指的是输入上采样后的输出通道数
        :param concat_channels: 指的是concat后的通道数
        :param out_channels: 指的是整个Up_Block的输出通道数
        :param upsample_method: 上采样方法 "conv_transpose代表转置卷积，bilinear代表双线性插值"
        :param up: 代表是否进行转置卷积，转置卷积会缩小特征图尺寸，如果不进行转置卷积，那么意味着收缩通道的下采样也需要取消掉
        """
        super(Up_Block_A, self).__init__()
        self.up = up
        if self.up == False:
            self.upsample = Basic_Conv_Block(in_channels, up_channels)
        else:
            if upsample_method == "transpose":
                self.upsample = nn.ConvTranspose2d(in_channels, up_channels, kernel_size=2, stride=2)
            elif upsample_method == "bilinear":
                self.upsample = nn.Sequential(
                    nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                    nn.Conv2d(in_channels, up_channels, kernel_size=1, stride=1)
                )
        self.conv1 = Basic_Conv_Block(concat_channels, out_channels)
        self.conv2 = Basic_Conv_Block(out_channels, out_channels)
        self.se = CSE_Block(out_channels)

    def forward(self, x, shortcut, enc_feature=None):
        x = self.upsample(x)
        # print(x.shape, shortcut.shape)
        if enc_feature is None:
            x = torch.cat([x, shortcut], dim=1)
        else:
            # print('up:', x.size(), shortcut.size(), enc_feature.size())
            x = torch.cat([x, shortcut, enc_feature], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        return x

class Up_Block_SP(nn.Module):
    def __init__(self, in_channels, up_channels, concat_channels, out_channels, upsample_method="transpose", up=True):
        """
        :param in_channels: 指的是输入的通道数
        :param up_channels: 指的是输入上采样后的输出通道数
        :param concat_channels: 指的是concat后的通道数
        :param out_channels: 指的是整个Up_Block的输出通道数
        :param upsample_method: 上采样方法 "conv_transpose代表转置卷积，bilinear代表双线性插值"
        :param up: 代表是否进行转置卷积，转置卷积会缩小特征图尺寸，如果不进行转置卷积，那么意味着收缩通道的下采样也需要取消掉
        """
        super(Up_Block_SP, self).__init__()
        self.up = up
        if self.up == False:
            self.upsample = Basic_Conv_Block(in_channels, up_channels)
        else:
            if upsample_method == "transpose":
                self.upsample = nn.ConvTranspose2d(in_channels, up_channels, kernel_size=2, stride=2)
            elif upsample_method == "bilinear":
                self.upsample = nn.Sequential(
                    nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                    nn.Conv2d(in_channels, up_channels, kernel_size=1, stride=1)
                )
        # self.conv1 = Basic_Conv_Block(concat_channels, out_channels)
        # self.conv2 = Basic_Conv_Block(out_channels, out_channels)
        self.conv1 = SPConv_3x3(concat_channels, out_channels)
        # print('concat_channel', concat_channels)
        self.conv2 = SPConv_3x3(out_channels, out_channels)

    def forward(self, x, shortcut, enc_feature=None):
        x = self.upsample(x)
        # print(x.shape, shortcut.shape)
        if enc_feature is None:
            x = torch.cat([x, shortcut], dim=1)
        else:
            # print('up:', x.size(), shortcut.size(), enc_feature.size())
            x = torch.cat([x, shortcut, enc_feature], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = Basic_Conv_Block(inplanes, outplanes, kernel_size=1, padding=0, dilation=dilations[0])
        self.aspp2 = Basic_Conv_Block(inplanes, outplanes, kernel_size=3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = Basic_Conv_Block(inplanes, outplanes, kernel_size=3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = Basic_Conv_Block(inplanes, outplanes, kernel_size=3, padding=dilations[3], dilation=dilations[3])
        self.avg_pool = nn.Sequential(
            nn.AvgPool2d((1, 1)),
            Basic_Conv_Block(inplanes, outplanes, kernel_size=1, padding=0)
        )

        self.project = nn.Sequential(
            nn.Conv2d(5 * outplanes, outplanes, 1, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU())

    def forward(self, x):
        x1 = self.aspp1(x)
        # print("x1 shape:", x1.shape)
        x2 = self.aspp2(x)
        # print("x2 shape:", x2.shape)
        x3 = self.aspp3(x)
        # print("x3 shape:", x3.shape)
        x4 = self.aspp4(x)
        # print("x4 shape:", x4.shape)
        x5 = self.avg_pool(x)
        # print("x5 shape:", x5.shape)
        x5 = F.interpolate(x5, x4.size()[2:], mode="bilinear", align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.project(x)
        return x


class CSE_Block(nn.Module):
    def __init__(self, inplanes, reduce_ratio=4):
        super(CSE_Block, self).__init__()
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


class Bottle2neck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, kernel_size=1,
                 stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)  # width*scale

        spx = torch.split(out, self.width, 1)  # [width, width, width, width]
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MSDMSA_bottle2neck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, kernel_size=56,
                 stype='normal'):
        super(MSDMSA_bottle2neck, self).__init__()
        width = int(math.floor(planes * (baseWidth / 128.0)))
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        # self.scale = scale
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.position_embed = build_position_encoding(mode='v2', hidden_dim=width)
        self.encoder_DeTrans = DeformableTransformer(d_model=width, dim_feedforward=width * 2,
                                                     dropout=0.1, activation='gelu', num_feature_levels=4, nhead=4,
                                                     num_encoder_layers=3, enc_n_points=4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def posi_mask(self, x):

        x_fea = []
        x_posemb = []
        masks = []
        for lvl, fea in enumerate(x):
            # if lvl > 1:
            x_fea.append(fea)
            x_posemb.append(self.position_embed(fea))
            masks.append(
                torch.zeros((fea.shape[0], fea.shape[2], fea.shape[3]), dtype=torch.bool).cuda())

        return x_fea, masks, x_posemb

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)  # width*scale
        out_features = []
        spx = torch.split(out, self.width, 1)  # [width, width, width, width]
        N, C, H, W = spx[0].size()
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
            out_features.append(sp)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
            out_features.append(spx[self.nums])
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)
            out_features.append(self.pool(spx[self.nums]))
            # print(self.pool(spx[self.nums]).size())
        # transformer
        x_fea, masks, x_posemb = self.posi_mask(out_features)
        x_trans = self.encoder_DeTrans(x_fea, masks, x_posemb)
        x_trans = x_trans.transpose(1, 2).view(N, C, self.nums + 1, H, W).flatten(1, 2)  # 2,3136,52 ->  2, 208 , 28, 28
        # reshape
        out = self.conv3(x_trans)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Unet_A_trans(nn.Module):
    def __init__(self, base_block, trans_block, layers, img_size=128, baseWidth=28, scale=4, num_classes=1):
        super(Unet_A_trans, self).__init__()
        # first encoder
        # self.base_model = base_model
        self.inplanes = 32
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(base_block, 32, layers[0])
        self.layer2 = self._make_layer(base_block, 64, layers[1])
        self.layer3 = self._make_layer(trans_block, 128, layers[2])
        self.layer4 = self._make_layer(trans_block, 256, layers[3], kernel_size=img_size // 8)
        self.layer5 = self._make_layer(trans_block, 256, layers[4], kernel_size=img_size // 16)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1)
        self.se1 = CSE_Block(64)
        self.se2 = CSE_Block(128)
        self.se3 = CSE_Block(256)
        self.se4 = CSE_Block(512)
        self.se5 = CSE_Block(512)

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
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

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
        en_x1 = self.se1(en_x1)
        pool_x1 = self.pool(en_x1)
        # print("unetA pool_x1 shape:", pool_x1.shape)
        en_x2 = self.layer2(pool_x1)
        en_x2 = self.se2(en_x2)
        pool_x2 = self.pool(en_x2)
        # print("unetA pool_x2 shape:", pool_x2.shape)
        en_x3 = self.layer3(pool_x2)
        en_x3 = self.se3(en_x3)
        pool_x3 = self.pool(en_x3)
        # print("unetA pool_x3 shape:", pool_x3.shape)
        en_x4 = self.layer4(pool_x3)
        en_x4 = self.se4(en_x4)
        pool_x4 = self.pool(en_x4)
        # print("unetA pool_x4 shape:", pool_x4.shape)
        en_x5 = self.layer5(pool_x4)
        en_x5 = self.se5(en_x5)
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

        encoder_features = [en_x1, en_x2, en_x3, en_x4]

        output = self.out_conv(de_x1)
        # output = F.sigmoid(output)
        # print("unetA output shape:", output.shape)

        return input, output, encoder_features
    
class Unet_trans(nn.Module):
    def __init__(self, base_block, trans_block, layers, img_size=128, baseWidth=28, scale=4, num_classes=1):
        super(Unet_trans, self).__init__()
        # first encoder
        # self.base_model = base_model
        self.inplanes = 32
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(base_block, 32, layers[0])
        self.layer2 = self._make_layer(base_block, 64, layers[1])
        # self.layer1 = conv_block(3, 64)
        # self.layer2 = conv_block(64, 128)
        self.layer3 = self._make_layer(trans_block, 128, layers[2])
        self.layer4 = self._make_layer(trans_block, 256, layers[3], kernel_size=img_size // 8)
        self.layer5 = self._make_layer(trans_block, 256, layers[4], kernel_size=img_size // 16)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1)
        # self.se1 = CSE_Block(64)
        # self.se2 = CSE_Block(128)
        # self.se3 = CSE_Block(256)
        # self.se4 = CSE_Block(512)
        # self.se5 = CSE_Block(512)

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
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


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
        # print("unetA pool_x2 shape:", pool_x2.shape, flush=True)
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

        # aspp_out = self.aspp(en_x5)
        # print("unetA aspp shape:", aspp_out.shape)

        de_x4 = self.up4(en_x5, en_x4)
        # print("unetA de_x4 shape:", de_x4.shape)
        de_x3 = self.up3(de_x4, en_x3)
        # print("unetA de_x3 shape:", de_x3.shape)
        de_x2 = self.up2(de_x3, en_x2)
        # print("unetA de_x2 shape:", de_x2.shape)
        de_x1 = self.up1(de_x2, en_x1)
        # print("unetA de_x1 shape:", de_x1.shape)

        encoder_features = [en_x1, en_x2, en_x3, en_x4]

        output = self.out_conv(de_x1)
        # output = F.sigmoid(output)
        # print("unetA output shape:", output.shape)

        return output

class Unet_A(nn.Module):
    def __init__(self, base_block, trans_block, layers, img_size=128, baseWidth=28, scale=4, num_classes=1):
        super(Unet_A, self).__init__()
        # first encoder
        # self.base_model = base_model
        self.inplanes = 32
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(base_block, 32, layers[0])
        self.layer2 = self._make_layer(base_block, 64, layers[1])
        self.layer3 = self._make_layer(base_block, 128, layers[2])
        self.layer4 = self._make_layer(base_block, 256, layers[3], kernel_size=img_size // 8)
        self.layer5 = self._make_layer(base_block, 256, layers[4], kernel_size=img_size // 16)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1)
        self.se1 = CSE_Block(64)
        self.se2 = CSE_Block(128)
        self.se3 = CSE_Block(256)
        self.se4 = CSE_Block(512)
        self.se5 = CSE_Block(512)

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
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

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
        en_x1 = self.se1(en_x1)
        pool_x1 = self.pool(en_x1)
        # print("unetA pool_x1 shape:", pool_x1.shape)
        en_x2 = self.layer2(pool_x1)
        en_x2 = self.se2(en_x2)
        pool_x2 = self.pool(en_x2)
        # print("unetA pool_x2 shape:", pool_x2.shape)
        en_x3 = self.layer3(pool_x2)
        en_x3 = self.se3(en_x3)
        pool_x3 = self.pool(en_x3)
        # print("unetA pool_x3 shape:", pool_x3.shape)
        en_x4 = self.layer4(pool_x3)
        en_x4 = self.se4(en_x4)
        pool_x4 = self.pool(en_x4)
        # print("unetA pool_x4 shape:", pool_x4.shape)
        en_x5 = self.layer5(pool_x4)
        en_x5 = self.se5(en_x5)
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

        encoder_features = [en_x1, en_x2, en_x3, en_x4]

        output = self.out_conv(de_x1)
        # output = F.sigmoid(output)
        # print("unetA output shape:", output.shape)

        return input, output, encoder_features

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
        self.up4 = Up_Block_A(512, 512, 1536, 512)
        self.up3 = Up_Block_A(512, 256, 768, 256)
        self.up2 = Up_Block_A(256, 128, 384, 128)
        self.up1 = Up_Block_A(128, 64, 192, 64)

        self.se1 = CSE_Block(64)
        self.se2 = CSE_Block(128)
        self.se3 = CSE_Block(256)
        self.se4 = CSE_Block(512)
        self.se5 = CSE_Block(512)

        # second out conv
        # self.out_conv = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        #     nn.Conv2d(in_channels=32, out_channels=out_c, kernel_size=(1, 1)),
        #     nn.Sigmoid()
        # )
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        # self.combine_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=out_c, kernel_size=1, padding=0),
        #     # nn.Sigmoid()
        # )

    def forward(self, input, output1, encoder_features):
        enc1, enc2, enc3, enc4 = encoder_features
        input_2 = input * output1
        # print("unetB input shape:", input_2.shape)
        en_x1 = self.layer1(input_2)  # 224,224,64
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
        de_x4 = self.up4(aspp_out, en_x4, enc4)
        de_x4 = self.se4(de_x4)
        # print("unetB de_x4 shape:", de_x4.shape)

        de_x3 = self.up3(de_x4, en_x3, enc3)
        de_x3 = self.se3(de_x3)
        # print("unetB de_x3 shape:", de_x3.shape)
        de_x2 = self.up2(de_x3, en_x2, enc2)
        de_x2 = self.se2(de_x2)
        # print("unetB de_x2 shape:", de_x2.shape)
        de_x1 = self.up1(de_x2, en_x1, enc1)
        de_x1 = self.se1(de_x1)
        # print("unetB de_x1 shape:", de_x1.shape)
        output2 = self.out_conv(de_x1)
        # output = torch.cat([output1, output2], dim=1)
        # output = self.combine_conv(output)

        return output2

class Unet_B_spp(nn.Module):
    def __init__(self, in_c, out_c, filter=64):
        super(Unet_B_spp, self).__init__()
        self.layer1 = conv_block(in_c, filter)  # 64
        self.layer2 = conv_block(filter, filter * 2)  # 128
        self.layer3 = conv_block(filter * 2, filter * 4)  # 256
        self.layer4 = conv_block(filter * 4, filter * 8)  # 512
        self.layer5 = conv_block(filter * 8, filter * 8)  # 512
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1)

        self.aspp = ASPP(512, 512)

        # first decoder
        self.up4 = Up_Block_SP(512, 512, 1536, 512)
        self.up3 = Up_Block_SP(512, 256, 768, 256)
        self.up2 = Up_Block_SP(256, 128, 384, 128)
        self.up1 = Up_Block_SP(128, 64, 192, 64)

        self.se1 = CSE_Block(64)
        self.se2 = CSE_Block(128)
        self.se3 = CSE_Block(256)
        self.se4 = CSE_Block(512)
        self.se5 = CSE_Block(512)

        # second out conv
        # self.out_conv = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        #     nn.Conv2d(in_channels=32, out_channels=out_c, kernel_size=(1, 1)),
        #     nn.Sigmoid()
        # )
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        # self.combine_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=out_c, kernel_size=1, padding=0),
        #     # nn.Sigmoid()
        # )

    def forward(self, input, output1, encoder_features):
        enc1, enc2, enc3, enc4 = encoder_features
        input_2 = input * output1
        # print("unetB input shape:", input_2.shape)
        en_x1 = self.layer1(input_2)  # 224,224,64
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
        de_x4 = self.up4(aspp_out, en_x4, enc4)
        de_x4 = self.se4(de_x4)
        # print("unetB de_x4 shape:", de_x4.shape)

        de_x3 = self.up3(de_x4, en_x3, enc3)
        de_x3 = self.se3(de_x3)
        # print("unetB de_x3 shape:", de_x3.shape)
        de_x2 = self.up2(de_x3, en_x2, enc2)
        de_x2 = self.se2(de_x2)
        # print("unetB de_x2 shape:", de_x2.shape)
        de_x1 = self.up1(de_x2, en_x1, enc1)
        de_x1 = self.se1(de_x1)
        # print("unetB de_x1 shape:", de_x1.shape)
        output2 = self.out_conv(de_x1)
        # output = torch.cat([output1, output2], dim=1)
        # output = self.combine_conv(output)

        return output2

class Unet_B_trans(nn.Module):
    def __init__(self, in_c, out_c, base_block, trans_block, layers, img_size=128, baseWidth=26, scale=4, num_classes=1, filter=64):
        super(Unet_B_trans, self).__init__()
        # first encoder
        # self.base_model = base_model
        self.inplanes = 32
        self.baseWidth = baseWidth
        self.scale = scale
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.relu = nn.ReLU(inplace=True)
        # self.layer1 = self._make_layer(base_block, 32, layers[0])
        # self.layer2 = self._make_layer(base_block, 64, layers[1])
        # self.layer3 = self._make_layer(base_block, 128, layers[2])
        self.layer1 = conv_block(in_c, filter)  # 64         
        self.layer2 = conv_block(filter, filter * 2)  # 128         
        self.layer3 = conv_block(filter * 2, filter * 4)  # 256
        self.layer4 = self._make_layer(trans_block, 256, layers[3], kernel_size=img_size // 8)
        self.layer5 = self._make_layer(trans_block, 256, layers[4], kernel_size=img_size // 16)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1)
        self.se1 = CSE_Block(64)
        self.se2 = CSE_Block(128)
        self.se3 = CSE_Block(256)
        self.se4 = CSE_Block(512)
        self.se5 = CSE_Block(512)

        # first aspp
        # self.aspp = ASPP(512, 512)

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
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.combine_conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=out_c, kernel_size=1, padding=0),  # nn.Sigmoid()
        )

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

    def forward(self, input, output1, encoder_features):
        enc1, enc2, enc3, enc4 = encoder_features
        input_2 = input * output1
        # print("unetB input shape:", input_2.shape)
        en_x1 = self.layer1(input_2)  # 224,224,64
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
        de_x4 = self.up4(en_x5, en_x4, enc4)
        de_x4 = self.se4(de_x4)
        # print("unetB de_x4 shape:", de_x4.shape)

        de_x3 = self.up3(de_x4, en_x3, enc3)
        de_x3 = self.se3(de_x3)
        # print("unetB de_x3 shape:", de_x3.shape)
        de_x2 = self.up2(de_x3, en_x2, enc2)
        de_x2 = self.se2(de_x2)
        # print("unetB de_x2 shape:", de_x2.shape)
        de_x1 = self.up1(de_x2, en_x1, enc1)
        de_x1 = self.se1(de_x1)
        # print("unetB de_x1 shape:", de_x1.shape)
        output2 = self.out_conv(de_x1)
        output = torch.cat([output1, output2], dim=1)
        output = self.combine_conv(output)

        return output

class ResDeTransDoubleUnet(nn.Module):
    def __init__(self, in_c, out_c, img_size=224):
        super(ResDeTransDoubleUnet, self).__init__()
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

class ResDeTransDoubleUnet_spp(nn.Module):
    def __init__(self, in_c, out_c, img_size=224):
        super(ResDeTransDoubleUnet_spp, self).__init__()
        # self.base_model = base_model
        self.unet1 = Unet_A_trans(Bottle2neck, MSDMSA_bottle2neck, [2, 2, 2, 2, 2], img_size)
        self.unet2 = Unet_B_spp(in_c, out_c, 64)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        input, output1, encoder_features = self.unet1(x)
        input2 = torch.sigmoid(output1)
        output2 = self.unet2(input, input2, encoder_features)
        # output1 = self.last_activation(output1)
        output2 = self.last_activation(output2)
        return input2, output2

class ResDoubleUnet(nn.Module):
    def __init__(self, in_c, out_c, img_size=224):
        super(ResDoubleUnet, self).__init__()
        # self.base_model = base_model
        self.unet1 = Unet_A(Bottle2neck, MSDMSA_bottle2neck, [2, 2, 2, 2, 2], img_size)
        self.unet2 = Unet_B(in_c, out_c, 64)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        input, output1, encoder_features = self.unet1(x)
        input2 = torch.sigmoid(output1)
        output2 = self.unet2(input, input2, encoder_features)
        # output1 = self.last_activation(output1)
        output2 = self.last_activation(output2)
        return input2, output2

class ResDeTransDoubleUnetV2(nn.Module):
    def __init__(self, in_c, out_c, img_size=224):
        super(ResDeTransDoubleUnet, self).__init__()
        # self.base_model = base_model
        self.unet1 = Unet_A_trans(Bottle2neck, MSDMSA_bottle2neck, [2, 2, 2, 2, 2], img_size)
        self.unet2 = Unet_B(in_c, out_c, 64)
        # self.unet2 = Unet_B_trans(in_c, out_c, Bottle2neck64, MSDMSA_bottle2neck, [2, 2, 2, 2, 2])
        self.last_activation = nn.Sigmoid()
        
    def forward(self, x):
        input, output1, encoder_features = self.unet1(x)
        input2 = torch.sigmoid(output1)
        output2 = self.unet2(input, input2, encoder_features)
        output1 = self.last_activation(output1)
        output2= self.last_activation(output2)
        return output1, output2

if __name__ == '__main__':
    model = ResDeTransDoubleUnet(3, 1, 224)
    # model = MSDMSA_bottle2neck(256, 256)
    x = torch.randn(2, 3, 224, 224)
    result = model(x)
    print(result[0].size(), result[1].size())
