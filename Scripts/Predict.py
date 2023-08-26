from PIL import Image



def pred(model, img_path, save_path, device):
    model.eval().to(device)
    input = Image.open(img_path)
    output = model(input)
    output.save(save_path)
    print('Finished! Save to: {}'.format(save_path))