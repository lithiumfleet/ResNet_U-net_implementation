import torch
from torch import device, nn 


def train_on_batch(loader, model, loss, optim, device):
	size = len(loader.dataset)
	model.train()
	for batch, (x, y) in enumerate(loader):
		x, y = x.to(device), y.to(device)
		pred = model(x)
		ls = loss(pred, y)
		ls.backward()
		optim.step()
		optim.zero_grad()
		if batch % 100 == 0:
			cur_loss, step = ls.item(), (batch+1)*len(x)
			print("loss={:>7f} step={}/{}".format(cur_loss, step, size))

def test(loader, model, loss):
	size = len(loader.dataset)
	numbatch = len(loader)
	model.eval()
	ls, acc = 0, 0
	with torch.no_grad():
		for x, y in loader:
			x, y = x.to(device), y.to(device)
			pred = model(x)
			ls += loss(pred, y).item()
			acc += (pred.argmax(1) == y).type(torch.float).sum().item()
	ls /= numbatch
	acc /= size
	acc *= 100
	print("[test]ls={:>7f} acc={:2f}%".format(ls, acc))

def train(model, train_dataloader, test_dataloader, loss_function, optimizer, epoch, device):
    for i in range(epoch):
        print("Epoch {}\n----------------------------".format(i))
        train_on_batch(train_dataloader, model, loss_function, optimizer, device)
        test(test_dataloader, model, loss_function)
    print("done")
