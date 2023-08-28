import torch
from torch import device, nn 
from .metrics import Accuracy

def train_one_epoch(trainloader, testloader, model, loss, optim, device, batch_size=16):
	size = len(trainloader.dataset)
	model.train()
	for batch, (x, y) in enumerate(trainloader):
		x, y = x.to(device), y.to(device)
		pred = model(x)
		ls = loss(pred, y)
		ls.backward()
		optim.step()
		optim.zero_grad()
		if batch % batch_size == 0 and batch > 0 :
			cur_loss, step, num_batch = ls.item(), (batch)*len(x), batch//batch_size
			print("[batch:{}] loss={:>7f} step={}/{}".format(num_batch,cur_loss, step, size))
			if num_batch > 0 and num_batch % 1 == 0:		# 测试
				test(testloader, model, loss, device) #TODO

def test(loader, model, loss, device):
	# size = len(loader.dataset)
	numtest = len(loader)
	model.eval()
	test_loss, acc = 0, 0	# get test loss
	with torch.no_grad():
		for i, (x, y) in enumerate(loader):
			x, y = x.to(device), y.to(device)
			pred = model(x).to(device)
			test_loss += loss(pred, y).item()
			acc += Accuracy(y, pred)
			if i % 100 == 0 and i > 0:
				break
	test_loss /= numtest
	acc = acc*100/numtest
	print("[test]    avg_loss={:>7f} acc={:3f}%".format(test_loss, acc))
		

def validate(model, val_dataloader, metrics, device):
	model.eval()
	metrics_vals = {loss.__name__ : 0 for loss in metrics}
	with torch.no_grad():
		for i, (x,y) in enumerate(val_dataloader):
			x, y = x.to(device), y.to(device)
			pred = model(x).to(device)
			for loss in metrics:
				metrics_vals[loss.__name__] += loss(y, pred)
			# if i==32:
				# break	# for debug
	lenth = len(val_dataloader.dataset)
	# lenth = 32
	avg_metrics_vals = {k:v.item()/lenth for k in metrics_vals.keys() for v in metrics_vals.values()} # type: ignore
	print(avg_metrics_vals)
		
				


def train(model, train_dataloader, test_dataloader, loss_function, optimizer, epoch, device, metrics=None):
    for i in range(epoch):
        print("Epoch {}\n".format(i), "-"*30)
        train_one_epoch(train_dataloader, test_dataloader, model, loss_function, optimizer, device)
    if metrics is not None:
        validate(model, test_dataloader, metrics, device)
    print("train done.")
    

	