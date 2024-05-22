import torch
import torch.nn as nn

from nets.CSPdarknet import C3, Conv, CSPDarknet
from nets.attention import cbam_block, eca_block, se_block, CA_Block

attention_block = [se_block, cbam_block, eca_block, CA_Block,]

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, pretrained=False,at=0):
        super(YoloBody, self).__init__()
        depth_dict          = {'n': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'n': 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        #-----------------------------------------------#
        self.backbone       = CSPDarknet(base_channels, base_depth, phi, pretrained)
            
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3         = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1    = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2         = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2    = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1  = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2  = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        # 80, 80, 256 => 80, 80, 3 * (5 + num_classes) => 80, 80, 3 * (4 + 1 + num_classes)
        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        # 40, 40, 512 => 40, 40, 3 * (5 + num_classes) => 40, 40, 3 * (4 + 1 + num_classes)
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        # 20, 20, 1024 => 20, 20, 3 * (5 + num_classes) => 20, 20, 3 * (4 + 1 + num_classes)
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)
        self.at = at
        if 1 <= self.at and self.at <= 4:
            self.feat1_att = attention_block[self.at - 1](int(wid_mul * 64) * 4)  # 256
            self.feat2_att = attention_block[self.at - 1](int(wid_mul * 64) * 8)  # 512
            self.feat3_att = attention_block[self.at - 1](int(wid_mul * 64) * 16)
            self.upsample5_att = attention_block[self.at - 1](int(wid_mul * 64) * 8)  ###
            self.upsample4_att = attention_block[self.at - 1](int(wid_mul * 64) * 4)  ###
            self.downsample3_att = attention_block[self.at - 1](int(wid_mul * 64) * 4)  ###
            self.downsample2_att = attention_block[self.at - 1](int(wid_mul * 64) * 8)  ###
            self.P4_att = attention_block[self.at - 1](int(wid_mul * 64) * 4)
            self.P5_att = attention_block[self.at - 1](int(wid_mul * 64) * 8)

    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone(x)

        if 1 <= self.at and self.at <= 6:
            feat1 = self.feat1_att(feat1)
            feat2 = self.feat2_att(feat2)
            feat3 = self.feat3_att(feat3)

        # 20, 20, 1024 -> 20, 20, 512
        P5          = self.conv_for_feat3(feat3)
        # 20, 20, 512 -> 40, 40, 512
        P5_upsample = self.upsample(P5)
        #attention
        if 1 <= self.at and self.at <= 4:
            P5_upsample = self.upsample5_att(P5_upsample)
        # 40, 40, 512 -> 40, 40, 1024

        P4          = torch.cat([P5_upsample, feat2], 1)
        # 40, 40, 1024 -> 40, 40, 512
        P4          = self.conv3_for_upsample1(P4)

        # 40, 40, 512 -> 40, 40, 256
        P4          = self.conv_for_feat2(P4)
        # 40, 40, 256 -> 80, 80, 256
        P4_upsample = self.upsample(P4)
        #attention
        if 1 <= self.at and self.at <= 4:
            P4_upsample = self.upsample4_att(P4_upsample)
        # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        P3          = torch.cat([P4_upsample, feat1], 1)
        # 80, 80, 512 -> 80, 80, 256
        P3          = self.conv3_for_upsample2(P3)
        
        # 80, 80, 256 -> 40, 40, 256
        P3_downsample = self.down_sample1(P3)
        if 1 <= self.at and self.at <= 4:
            P3_downsample = self.downsample3_att(P3_downsample)
        if 1 <= self.at and self.at <= 4:
            P4 = self.P4_att(P4)

        # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
        P4 = torch.cat([P3_downsample, P4], 1)
        # 40, 40, 512 -> 40, 40, 512
        P4 = self.conv3_for_downsample1(P4)

        # 40, 40, 512 -> 20, 20, 512
        P4_downsample = self.down_sample2(P4)
        if 1 <= self.at and self.at <= 4:
            P4_downsample = self.downsample2_att(P4_downsample)
        if 1 <= self.at and self.at <= 4:
            P5 = self.P5_att(P5)
        # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        P5 = torch.cat([P4_downsample, P5], 1)
        # 20, 20, 1024 -> 20, 20, 1024
        P5 = self.conv3_for_downsample2(P5)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,80,80)
        #---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,40,40)
        #---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,20,20)
        #---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)
        return out0, out1, out2

