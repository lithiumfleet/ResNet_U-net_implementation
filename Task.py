from lzma import MODE_NORMAL
import torch
from torch import device, nn 
from ResUnetModels import ResNet, UNet, PlainNet
from dataset import VOC
from torch.utils.data.dataloader import DataLoader
from Scripts import Predict, Train, metrics

class Task():
    def __init__(self, model_name:str, model:str, epoch:int, input_img:str, mode:str='train', save_path='./result/img', lr:float=1e-3, metrics=None):
        self.model_name = model_name
        self.model = self.get_model(model)
        self.mode = mode
        self.epoch = epoch
        self.train_dataloader = VOC.get_VOC_loader(self.model_name, 'train')
        self.test_dataloader = VOC.get_VOC_loader(self.model_name, 'val')
        self.input_img = input_img
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.save_path = save_path + '/' + self.model_name + '.pt'
        self.loss = nn.BCELoss()#nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.metrics = metrics
        
    def train(self):
        return Train.train(
            self.model, 
            self.train_dataloader,
            self.test_dataloader,
            self.loss,
            self.optim,
            self.epoch,
            self.device,
            self.metrics
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
            if self.model_name == 'resnet18': return ResNet.ResNet18(classes=20)
            elif self.model_name == 'resnet34': return ResNet.ResNet34(classes=20)
            elif self.model_name == 'plainnet18': return PlainNet.PlainNet18(classes=20)
            elif self.model_name == 'plainnet34': return PlainNet.PlainNet34(classes=20)
            else: return UNet.Unet()
    
    def save_model(self):
        torch.save(self.model, self.save_path)
        print('model saved.')


if __name__ == "__main__":
    t = Task(
            model_name = 'resnet18',
            model = 'none',
            epoch = 0,
            input_img = r'D:',
            mode = 'train',
            metrics=[metrics.F1Measure, metrics.Precision],
            save_path=r'D:'
        )
    # t.train()
    # t.save_model()
    # t.predict()
    