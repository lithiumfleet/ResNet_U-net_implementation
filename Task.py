from lzma import MODE_NORMAL
import torch
from torch import device, nn 
from ResUnetModels import ResNet, UNet, PlainNet
from dataset import VOC
from torch.utils.data.dataloader import DataLoader
from Scripts import Predict, Train

class Task():
    def __init__(self, model_name:str, model:str, epoch:int, input_img:str, mode:str='train', save_path='./result/img', lr:float=1e-3):
        self.model_name = model_name
        self.model = self.get_model(model)
        self.mode = mode
        self.epoch = epoch
        self.train_dataloader = VOC.get_VOC_loader(self.model_name, self.mode)
        self.test_dataloader = VOC.get_VOC_loader(self.model_name, self.mode)
        self.input_img = input_img
        self.device = 'GPU' if torch.cuda.is_available() else 'CPU'
        self.save_path = save_path
        self.loss = nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(self.model.parameters(), lr=lr)
        
    def train(self):
        return Train.train(
            self.model, 
            self.train_dataloader,
            self.test_dataloader,
            self.loss,
            self.optim,
            self.epoch,
            self.device
        )
    
    def predict(self):
        return Predict.pred(
            self.model,
            self.input_img,
            self.save_path,
            self.device
        )
    
    def get_model(self, model_path):
        if model_path != 'none':
            return torch.load(model_path)
        else:
            if self.model_name == 'resnet18': return ResNet.ResNet18()
            elif self.model_name == 'resnet34': return ResNet.ResNet34()
            elif self.model_name == 'plainnet18': return PlainNet.PlainNet18()
            elif self.model_name == 'plainnet34': return PlainNet.PlainNet34()
            else: return UNet.Unet()


if __name__ == "__main__":
    t = Task(
            model_name = 'resnet18',
            model = 'none',
            epoch = 1,
            input_img = 'none',
            mode = 'train'
        )
    t.train()
    # t.train()
    # t.predict()