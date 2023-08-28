import torch
from torch import nn 

class Block(nn.Module):
    def __init__(self, inch, outch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inch, outch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outch),
            nn.ReLU(),
            nn.Conv2d(outch, outch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outch),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.block(x)
    


class PlainNet(nn.Module):
    def __init__(self, *blocks, classes=1000):
        super(PlainNet, self).__init__()
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
        layer = nn.Sequential(
            Block(inch, outch, stride=stride),
        )
        for i in range(num):
            layer.append(Block(outch, outch))
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

def PlainNet18(classes=1000):
    return PlainNet(2,2,2,2, classes=classes)

def PlainNet34(classes=1000):
    return PlainNet(3,4,6,3, classes=classes)

# debug
if __name__ == '__main__':
    p = PlainNet18()
    pp = PlainNet34()
    tests = torch.rand(64,3,224,224)
    # print(p)
    # print('='*30)
    # print(pp)
    print(pp(tests))


