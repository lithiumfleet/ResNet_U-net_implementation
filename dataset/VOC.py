import torchvision
from torch.utils.data import dataloader
from torchvision.datasets import VOCDetection, VOCSegmentation


def get_dataset_path(dataset_name:str, mode:str):
    path = './dataset/data/{}/'.format(dataset_name)
    return path


def to_loader(dataset):
    loader = dataloader.DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
    )
    return loader




def get_VOC_loader(model_name:str, mode:str):
    if model_name.startswith('resnet') or model_name.startswith('plainnet'):
        dataset = VOCDetection(
            get_dataset_path('VOCDetection', mode),
            '2012',
            mode,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )
        return to_loader(dataset)
    else:
        dataset = VOCSegmentation(
            get_dataset_path('VOCSegmentation', mode),
            '2012',
            mode,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )
        return to_loader(dataset)
        
