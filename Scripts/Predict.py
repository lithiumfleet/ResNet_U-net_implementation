from PIL import Image
import torch, os
from time import time
from torchvision.transforms import ToPILImage, PILToTensor


def pred(model, img_path, save_path, device):
    model.eval().to(device)
    input = PILToTensor()(Image.open(img_path)).unsqueeze(0).to(torch.float).to(device)
    output = model(input)
    name = '/output_{}.pt'.format(int(time()))
    save_path += name
    if save_path[0] == '.':
        save_path = save_path[1:]
        wkspace = os.getcwd()
        save_path = wkspace + save_path
    with open(save_path, 'w'):
        pass
    torch.save(output, save_path)
    print('Finished. Save to: {}'.format(save_path))