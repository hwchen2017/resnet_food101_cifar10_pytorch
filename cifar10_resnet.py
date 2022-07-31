import os, sys
import torch
import argparse
import torch.nn as nn
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from template import AverageMeter


'''
class CNNModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
			nn.ReLU())
		self.conv2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(), 
			nn.MaxPool2d(2, 2))
		self.conv3 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1), 
			nn.ReLU())
		self.conv4 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1), 
			nn.ReLU(), 
			nn.MaxPool2d(2, 2))
		self.conv5 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1), 
			nn.ReLU())
		self.conv6 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1), 
			nn.ReLU(), 
			nn.MaxPool2d(2, 2))

		self.linear_layer = nn.Sequential(
			nn.Flatten(), 
			nn.Linear(256*4*4, 1024), 
			nn.ReLU(), 
			nn.Linear(1024, 512), 
			nn.ReLU(), 
			nn.Linear(512, 10))

	def forward(self, x):
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.conv4(out)
		out = self.conv5(out)
		out = self.conv6(out)
		out = self.linear_layer(out)

		return out
'''



class Resnet9(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(64),
			nn.ReLU())
		self.conv2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(128),
			nn.ReLU(), 
			nn.MaxPool2d(2))
		self.conv3 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(128),
			nn.ReLU())
		self.conv4 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(128),
			nn.ReLU())
		self.conv5 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(256),
			nn.ReLU(), 
			nn.MaxPool2d(2))
		self.conv6 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(512),
			nn.ReLU(), 
			nn.MaxPool2d(2))
		self.conv7 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(512),
			nn.ReLU())
		self.conv8 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(512),
			nn.ReLU())
		self.classifier = nn.Sequential(
			nn.MaxPool2d(4), 
			nn.Flatten(), 
			nn.Dropout(0.2),
			nn.Linear(512, 10))

	def forward(self, x):
		out = self.conv1(x)
		out = self.conv2(out)
		res = out
		out = self.conv3(out)
		out = self.conv4(out) + res
		out = self.conv5(out)
		out = self.conv6(out)
		res = out
		out = self.conv7(out)
		out = self.conv8(out) + out
		out = self.classifier(out)

		return out	



def training(train_loader, model, optimizer, sched, print_freq=50):

	model.train()

	loader_size = len(train_loader)

	losses = AverageMeter()

	for num, batch in enumerate(train_loader):
		
		optimizer.zero_grad()
		images, labels = batch
		images = images.cuda()
		labels = labels.cuda()
		out = model(images)
		loss = nn.functional.cross_entropy(out, labels)

		loss.backward()
		optimizer.step()

		sched.step()

		losses.update(loss.item())

		if(num % print_freq == 0):
			print("Progress: %.2f%%, loss: %.3f" % (num/loader_size*100, loss.item()) )
			# losses.update(loss)

	print("Average loss: %.2f" % losses.avg)    
	return losses.avg


parser = argparse.ArgumentParser(description = 'Pytorch Resnet Training for CIFAR10')
parser.add_argument('--epoch', help='Number of training epoches', type=int, default=40)
parser.add_argument('--batchsize', help = 'Size of training batch', type = int, default = 512)
parser.add_argument('--lr', help = 'Laraning rate', type = float, default = 0.005)
parser.add_argument('--weight_decay', help = "Weight decay", type = float, default = 1e-4)
parser.add_argument('--workers', help = 'Number of worker', type = int, default = 4)
parser.add_argument('--dataset', help = 'The path of training data', type = str, default = './cifar10')
# parser.add_argument('--pretrained', help = 'Using imagenet pretrained model', type = bool, default = True)
parser.add_argument('--save_checkpoint', help = 'True for saving checkpoint during training', type = bool, default = True)
parser.add_argument('--checkpoint_path', help = 'The path for checkpoint', type = str, default = './best_checkpoint.pth')
parser.add_argument('--print_freq', help = 'Print frequency', type = int, default = 50)
parser.add_argument('--evaluate', help = 'Evaluate mode', type = bool, default = False)

args = parser.parse_args()



train_tfms = transforms.Compose([
	transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
	transforms.RandomHorizontalFlip(), 
	transforms.ToTensor(), 
	transforms.Normalize( [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010], inplace=True)
	])

valid_tfms = transforms.Compose([
	transforms.ToTensor(), 
	transforms.Normalize( [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
	])


# dataset = ImageFolder(data_dir + '/train', transform = ToTensor())



data_dir = args.dataset 
train_ds = ImageFolder(data_dir+'/train', train_tfms)
valid_ds = ImageFolder(data_dir+'/test', valid_tfms)

random_seed = 42
torch.manual_seed(random_seed)

batch_size = args.batchsize 
num_epochs = args.epoch
save_checkpoint = args.save_checkpoint
print_freq = args.print_freq


train_loader = DataLoader(train_ds, batch_size, shuffle = True, num_workers = args.workers, pin_memory = True)
valid_loader = DataLoader(valid_ds, batch_size*2, num_workers = args.workers, pin_memory = True)


max_lr = args.lr
# model = CNNModel()
model = Resnet9() 
model = model.cuda() 


optimizer = torch.optim.Adam(model.parameters(), max_lr, weight_decay = args.weight_decay)
# sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [8, 16, 24, 32], gamma = 0.2 )
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=num_epochs, steps_per_epoch=len(train_loader))
# optimizer = torch.optim.Adam(model.parameters(), 0.001)
# criterion = nn.functional.cross_entropy().cuda()

best_acc = 0.0
start_epoch = 0


checkpoint_path = args.checkpoint_path


if(os.path.exists(checkpoint_path)):
	print("Saved model exists")
	checkpoint = torch.load(checkpoint_path)
	loss = checkpoint['loss']
	start_epoch = checkpoint['epoch'] + 1
	best_acc = checkpoint['accuracy']
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	sched.load_state_dict(checkpoint['scheduler_state_dict'])

	print("Accuracy of saved model is: %.2f%%" % best_acc)



if(args.evaluate == True):

	with torch.no_grad():
		
		acc = AverageMeter()

		model.eval()

		for batch in valid_loader:
			images, labels = batch
			images = images.cuda()
			labels = labels.cuda()
			out = model(images)
			loss = nn.functional.cross_entropy(out, labels)
			_, preds = torch.max(out, dim = 1)
			current_acc = torch.tensor(torch.sum(preds == labels).item()/len(preds)).item()

			acc.update(current_acc)

		print("Accuracy on test data: %.2f%%" % (acc.avg*100.0))

	sys.exit()



fp = open("cifar10_training.txt", "w")



for epoch in range(start_epoch, num_epochs): 

	print("Epoch %d: " % epoch)

	train_loss = training(train_loader, model, optimizer, sched, print_freq)

	# print("Current learning rate: ", sched.get_last_lr())

	# sched.step()		

	with torch.no_grad():
		
		acc = AverageMeter()
		losses = AverageMeter()
        
		model.eval()

		for batch in valid_loader:
			images, labels = batch
			images = images.cuda()
			labels = labels.cuda()
			out = model(images)
			loss = nn.functional.cross_entropy(out, labels)
			_, preds = torch.max(out, dim = 1)
			current_acc = torch.tensor(torch.sum(preds == labels).item()/len(preds)).item()
            
			losses.update(loss.item())
			acc.update(current_acc)

		print("Accuracy on test data: %.2f%%, loss: %.2f" % (acc.avg*100.0, losses.avg))
        
		fp.write("%d %.2f %.2f %.2f\n" %(epoch, train_loss, losses.avg, acc.avg) )
        
		if(acc.avg > best_acc and (save_checkpoint == True or epoch == num_epochs - 1)):
			best_acc = acc.avg
			torch.save({
				'epoch': epoch,
				'accuracy': (acc.avg * 100.0), 
				'loss': loss.item(),  
				'model_state_dict': model.state_dict(), 
				'optimizer_state_dict': optimizer.state_dict(), 
				'scheduler_state_dict': sched.state_dict()
				}, checkpoint_path)


            
fp.close()






