import torch
import torch.nn as nn
import torchvision.models as models


def Deconv(n_input, n_output, k_size=4, stride=2, padding=1):
    Tconv = nn.ConvTranspose2d(
        n_input, n_output,
        kernel_size=k_size,
        stride=stride, padding=padding,
        bias=False)
    block = [
        Tconv,
        nn.BatchNorm2d(n_output),
        nn.LeakyReLU(inplace=True),
    ]
    return nn.Sequential(*block)
        

def Conv(n_input, n_output, k_size=4, stride=2, padding=0, bn=False, dropout=0):
    conv = nn.Conv2d(
        n_input, n_output,
        kernel_size=k_size,
        stride=stride,
        padding=padding, bias=False)
    block = [
        conv,
        nn.BatchNorm2d(n_output),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(dropout)
    ]
    return nn.Sequential(*block)


class Unet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        
        self.resnet = models.resnet50(pretrained=True)
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        # get some layer from resnet to make skip connection
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        
        # convolution layer, use to reduce the number of channel => reduce weight number
        self.conv_5 = Conv(2048, 512, 1, 1, 0)
        self.conv_4 = Conv(1536, 512, 1, 1, 0)
        self.conv_3 = Conv(768, 256, 1, 1, 0)
        self.conv_2 = Conv(384, 128, 1, 1, 0)
        self.conv_1 = Conv(128, 64, 1, 1, 0)
        # self.conv_0 = Conv(32, 1, 3, 1, 1)
        
        self.out = nn.Conv2d(32, n_classes, 1)
        
        # deconvolution layer
        self.deconv4 = Deconv(512, 512, 4, 2, 1)
        self.deconv3 = Deconv(512, 256, 4, 2, 1)
        self.deconv2 = Deconv(256, 128, 4, 2, 1)
        self.deconv1 = Deconv(128, 64, 4, 2, 1)
        self.deconv0 = Deconv(64, 32, 4, 2, 1)

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip_1 = x
        
        x = self.maxpool(x)
        x = self.layer1(x)
        skip_2 = x

        x = self.layer2(x)
        skip_3 = x
        x = self.layer3(x)
        skip_4 = x
        
        x5 = self.layer4(x)
        x5 = self.conv_5(x5)
        
        x4 = self.deconv4(x5)
        x4 = torch.cat([x4, skip_4], dim=1)
        x4 = self.conv_4(x4)
        
        x3 = self.deconv3(x4)
        x3 = torch.cat([x3, skip_3], dim=1)
        x3 = self.conv_3(x3)
        
        x2 = self.deconv2(x3)
        x2 = torch.cat([x2, skip_2], dim=1)
        x2 = self.conv_2(x2)
        
        x1 = self.deconv1(x2)
        x1 = torch.cat([x1, skip_1], dim=1)
        x1 = self.conv_1(x1)
        
        x0 = self.deconv0(x1)
        x0 = self.out(x0)
        
        # x0 = self.sigmoid(x0)
        return x0