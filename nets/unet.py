import torch
import torch.nn as nn

from nets.repvgg import create_RepVGG_B2
from nets.resnet import resnet50
from nets.vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        elif backbone == "repvgg":
            self.repvgg = create_RepVGG_B2(deploy=False, pretrained=pretrained)
            in_filters = [192, 416, 832, 3200]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50' or backbone == 'repvgg':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
            # self.up_conv = nn.Sequential(
            #     nn.UpsamplingBilinear2d(scale_factor=2),
            #     nn.Conv2d(130, 130, kernel_size=3, padding=1),
            #     nn.ReLU(),
            #     nn.Conv2d(130, 130, kernel_size=3, padding=1),
            #     nn.ReLU(),
            # )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        # self.pixel_final = nn.Conv2d(130, num_classes, 1)

        # self.pixel1 = nn.PixelShuffle(1)
        # self.pixel2 = nn.PixelShuffle(2)
        # self.pixel3 = nn.PixelShuffle(4)
        # self.pixel4 = nn.PixelShuffle(8)
        # self.pixel5 = nn.PixelShuffle(16)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        elif self.backbone == "resnet34":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        elif self.backbone == "resnet18":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        elif self.backbone == "repvgg":
            [feat1, feat2, feat3, feat4, feat5] = self.repvgg.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        # up5_pixel = self.pixel5(feat5)
        # up4_pixel = self.pixel4(up4)
        # up4_pixel = torch.concat([up5_pixel, up4_pixel], dim=1)
        #
        # up3_pixel = self.pixel3(up3)
        # up3_pixel = torch.concat([up4_pixel, up3_pixel], dim=1)
        #
        # up2_pixel = self.pixel2(up2)
        # up2_pixel = torch.concat([up3_pixel, up2_pixel], dim=1)
        #
        # up1_pixel = self.pixel1(up1)
        # up1_pixel = torch.concat([up2_pixel, up1_pixel], dim=1)

        if self.up_conv != None:
            up1 = self.up_conv(up1)
            # up1_pixel = self.up_conv(up1_pixel)

        final = self.final(up1)
        # final = self.pixel_final(up1_pixel)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet34":
            for param in self.resnet.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet18":
            for param in self.resnet.parameters():
                param.requires_grad = False
        elif self.backbone == "repvgg":
            for param in self.repvgg.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet34":
            for param in self.resnet.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet18":
            for param in self.resnet.parameters():
                param.requires_grad = True
        elif self.backbone == "repvgg":
            for param in self.repvgg.parameters():
                param.requires_grad = True
