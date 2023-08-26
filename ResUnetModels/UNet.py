import torch
from torch import nn 

class convBlock(nn.Module):
    def __init__(self, inch, outch):
        super(convBlock, self).__init__()
        self.outch = outch
        self.block = nn.Sequential(
            nn.Conv2d(inch, outch, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(outch),
            nn.ReLU(inplace=True),
            nn.Conv2d(outch, outch, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(outch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        assert x.shape[1] == self.outch
        return x


class downSamp(nn.Module):
    def __init__(self, inch, outch):
        super(downSamp, self).__init__()
        self.outch = outch
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            convBlock(inch, outch)
        )

    def forward(self, x):
        x = self.block(x)
        assert x.shape[1] == self.outch
        return x


class upSamp(nn.Module):
    """
    upsampling and conv
    param a: need to upsamp
    param b: need to concat
    """
    def __init__(self, inch, outch):
        super(upSamp, self).__init__()
        self.outch = outch
        self.up = nn.ConvTranspose2d(inch, inch//2, kernel_size=2, stride=2)
        self.conv = convBlock(inch, outch) # 没有采用bilinear: 1.源码 2.bi方法有一定缺陷

    def corp_cat(self, a, b):
        dx, dy = a.shape[2]-b.shape[2], a.shape[3]-b.shape[3]
        pad_b = torch.nn.functional.pad(b,[dx//2, dx-dx//2, dy//2, dy-dy//2]) # padding issue: https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        ab = torch.cat([a,pad_b], dim=1)
        return ab
        

    def forward(self, a, b):
        b = self.up(b)
        ab = self.corp_cat(a, b)
        ab = self.conv(ab)
        assert ab.shape[1] == self.outch
        return ab


class UNet(nn.Module):
    def __init__(self, channels, classes):
        super(UNet, self).__init__()
        self.channels = channels
        self.classes = classes

        self.input_layer = convBlock(channels, 64)
        self.down1 = downSamp(64, 128)
        self.down2 = downSamp(128, 256)
        self.down3 = downSamp(256, 512)
        self.down4 = downSamp(512, 1024)
        self.up1 = upSamp(1024, 512)
        self.up2 = upSamp(512, 256)
        self.up3 = upSamp(256, 128)
        self.up4 = upSamp(128, 64)
        self.output_layer = nn.Conv2d(64,classes,kernel_size=1)

    def forward(self, x):
        x1 = self.input_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        pred = self.output_layer(x)
        return pred


def Unet():
    return UNet(3, 21)

# debug
if __name__ == "__main__":
    # b = convBlock(1,64)
    # d = downSamp(64,64)
    # u = upSamp(256,256)
    n = UNet(3,120)
    # testb = torch.rand(8,1,572,572)
    # testd = torch.rand(8,64,568,568)
    # testua = torch.rand(1,128,280,280)
    # testub = torch.rand(1,256,100,100)
    testn = torch.rand(1,3,572,572)
    # print(b(testb).shape)
    # print(d(testd).shape)
    # print(u(testua, testub).shape)
    print(n(testn).shape)