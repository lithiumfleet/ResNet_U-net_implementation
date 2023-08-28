import torch
from torch import nn

class ResidualBlock(nn.Module):
    """
    residual block 
    """
    def __init__(self, inch, outch, shortcut=None, stride=1):
        """
        1. without shortcut, inch == outch. 
        2. parameter stride use for first conv
        """
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inch, outch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outch),
            nn.ReLU(),
            nn.Conv2d(outch, outch, 3, stride=1, padding=1, bias=False), # 啊???????
            nn.BatchNorm2d(outch)
        )
        self.shortcut = shortcut
    
    def forward(self, input):
        output = self.block(input)
        residual = input if self.shortcut is None else self.shortcut(input)
        output += residual
        return torch.relu(output)

# class BottleNeck(nn.Module):
#     def __init__(self, inch, outch, downsampling=False, stride=1, expansion=4):
#         super(BottleNeck, self).__init__()
#         self.expansion, self.downsampling = expansion, downsampling

#         self.block = nn.Sequential(
#             nn.Conv2d(inch, outch, kernel_size=1, stride=stride, bias=False),
#             nn.BatchNorm2d(outch),
#             nn.ReLU(True),
#             nn.Conv2d(inch, outch, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(outch),
#             nn.ReLU(True),
#             nn.Conv2d(inch, outch*self.expansion, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(outch*self.expansion)
#         )



class ResNet(nn.Module):
    def __init__(self, *blocks, classes=1000):
        super(ResNet,self).__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.layer1 = self.get_layer(64, 64, blocks[0])
        self.layer2 = self.get_layer(64, 128, blocks[1], stride=2)
        self.layer3 = self.get_layer(128, 256, blocks[2], stride=2)
        self.layer4 = self.get_layer(256, 512, blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fcnet = nn.Linear(512, classes)
        self.logist = nn.Sigmoid()
        
    def get_layer(self, inch, outch, num, stride=1):
        """
        num: 'num of layers in a block'
        """
        if stride != 1:
            layer_shortcut = nn.Sequential( # 层间shortcut, 使用卷积结果作为x
                nn.Conv2d(inch, outch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outch),
                nn.ReLU()
            )
        else:
            layer_shortcut = None

        layer = nn.Sequential()
        layer.append(ResidualBlock(inch, outch, shortcut=layer_shortcut, stride=stride))
        for i in range(1, num):
            layer.append(ResidualBlock(outch, outch))
        return layer
    
    def forward(self, x):
        x = self.preprocess(x)
        # print('preprocess:',x.shape)
        x = self.layer1(x)
        # print('layer1:',x.shape)
        x = self.layer2(x)
        # print('layer2:',x.shape)
        # assert x.shape[2] == 28
        x = self.layer3(x)
        # print('layer3:',x.shape)
        x = self.layer4(x)
        # print('layer4:',x.shape)
        x = self.avg_pool(x)
        x = self.flatten(x)
        # print('flatten:',x.shape)
        # assert x.shape[1] == 512
        x = self.fcnet(x)
        return self.logist(x)

def ResNet18(classes=1000):
    """
    for imgnet, 224*224
    """
    return ResNet(2,2,2,2, classes=classes)

def ResNet34(classes=1000):
    """
    for imgnet, 224*224
    """
    return ResNet(3,4,6,3, classes=classes)





# for debug
if __name__ == '__main__':
    r = ResNet18(classes=20)
    rr = ResNet34()
    # b = ResidualBlock(3,2)

    # print(r)
    # print('='*20)
    # print(rr)
    # test = torch.rand(1,3,224,224)
    tests = torch.rand(64,3,100,356)
    print(r(tests)[0])
    # res = b(test)
    # print(res.shape)