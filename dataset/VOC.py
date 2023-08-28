import torchvision
from torch import zeros
from torch.utils.data import dataloader
from torchvision.datasets import VOCDetection, VOCSegmentation
from PIL import Image 
from .get_dict import Cls2Vec

# def reshape(img):
#     return img.resize((224,224))

def get_dataset_path(dataset_name:str, mode:str):
    path = './dataset/data/{}'.format(dataset_name)
    return path

# def collate(batch:list):
#     new_batch = []
#     for x, annotation in batch:
#         y = annotation['annotation']['object'][0]['name'][0]
#         new_batch.append((x,y))
#     return new_batch

def to_loader(dataset):
    loader = dataloader.DataLoader(
        dataset=dataset,
        batch_size=1,  #FIXME: batchsize
        shuffle=True
        # collate_fn=collate
    )
    return loader

compose =  torchvision.transforms.Compose([
    # torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((224,224)),
    # torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor()
    ])


def preprocess(img, xml):
    img = compose(img)
    obj = xml['annotation']['object']
    y = zeros(20) # 采用硬编码, 因为这里就是VOC, 只有20个类和背景
    for i, label in enumerate(obj):
        name = label['name']
        y += Cls2Vec[name]
    return img, y


def get_VOC_loader(model_name:str, mode:str):
    if model_name.startswith('resnet') or model_name.startswith('plainnet'):
        dataset = VOCDetection(
            get_dataset_path('VOCDetection', mode),
            '2012',
            mode,
            download=False,
            transforms=preprocess
        )
        return to_loader(dataset)
    else:
        dataset = VOCSegmentation(
            get_dataset_path('VOCSegmentation', mode),
            '2012',
            mode,
            download=False,
            transforms=preprocess
        )
        return to_loader(dataset)
        
if __name__ == "__main__":
    dl = get_VOC_loader('resnet', 'train')
    # print(dl.dataset)
    for a,b in dl:
        print(a.shape)
        print(b)
        # print(y['annotation']['object'][0]['name'][0])