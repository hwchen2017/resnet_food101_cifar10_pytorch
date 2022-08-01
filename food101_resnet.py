import os, sys
import torch 
import torch.nn as nn
import numpy as np 
import torchvision
import torch.utils.data as data
from torchvision import datasets, transforms
import resnet
from PIL import Image
from template import AverageMeter
import argparse


class food101(torch.utils.data.Dataset):
	def __init__(self, dataroot, split, transform):
		super(food101, self).__init__()
		self.dataroot = dataroot
		self.img_folder = os.path.join(dataroot, 'images')
		if split =='train':
			train_txt_path = os.path.join(self.dataroot, 'meta/train.txt')
			with open(train_txt_path) as f:
				lines = f.readlines()
		elif split in ['val', 'test']:
			test_txt_path = os.path.join(self.dataroot, 'meta/test.txt')
			with open(test_txt_path) as f:
				lines = f.readlines()
		else:
			raise ValueError('split = %s is not supported' % split)

		self.lines = lines
		self.transform =transform

		# read labels
		# labels = []
		with open(os.path.join(self.dataroot, 'meta/classes.txt')) as f:
			labels = f.readlines()
		labels = [label.strip() for label in labels]
		self.labels = labels

	def __len__(self):
		return len(self.lines)

	def __getitem__(self, x):
		line = self.lines[x].strip()
		dir_name = os.path.dirname(line)
		label = self.labels.index(dir_name)
		img_path = os.path.join(self.img_folder, line + '.jpg')
		img = Image.open(img_path).convert('RGB')
		return self.transform(img),label

	
def accuracy(output, target, topk=(1,)):
	
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)
		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		
		correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)

		res = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(1.0 / batch_size))
		return res


def training(train_loader, model, optimizer, print_freq = 50):

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

		losses.update(loss.item())

		# sched.step()
		if(num % print_freq == 0):
			print("Progress: %.2f%%, loss: %.3f" % (num/loader_size*100, loss.item()), flush = True )
			# print("Progress: "num, loss.item(), flush=True)

	print("Average loss: %.2f" % losses.avg)
	return losses.avg



parser = argparse.ArgumentParser(description = 'Pytorch Resnet Training for Food101')
parser.add_argument('--epoch', help='Number of training epoches', type=int, default=40)
parser.add_argument('--arch', help='Architecture of Resnet Model: resnet34, resnet50, resnet101', type=str, default='resnet50')
parser.add_argument('--batchsize', help = 'Size of training batch', type = int, default = 128)
parser.add_argument('--lr', help = 'Laraning rate', type = float, default = 0.01)
parser.add_argument('--weight_decay', help = "Weight decay", type = float, default = 1e-4)
parser.add_argument('--workers', help = 'Number of worker', type = int, default = 4)
parser.add_argument('--dataset', help = 'The path of training data', type = str, default = './food-101')
parser.add_argument('--pretrained', help = 'Using imagenet pretrained weights', type = bool, default = True)
parser.add_argument('--save_checkpoint', help = 'True for saving checkpoint during training', type = bool, default = True)
parser.add_argument('--checkpoint_path', help = 'The path for checkpoint', type = str, default = './best_checkpoint.pth')
parser.add_argument('--print_freq', help = 'Print frequency', type = int, default = 50)
parser.add_argument('--evaluate', help = 'Evaluate mode', type = bool, default = False)

args = parser.parse_args()




train_transforms = transforms.Compose([
	transforms.Resize(224), 
	transforms.CenterCrop(224), 
	transforms.RandomHorizontalFlip(),
	transforms.RandomVerticalFlip(),
	transforms.RandomRotation(45),
	transforms.RandomAffine(45),
	transforms.ColorJitter(),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

valid_transforms = transforms.Compose([
	transforms.Resize(256), 
	transforms.CenterCrop(224), 
	transforms.ToTensor(), 
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])



random_seed = 42
torch.manual_seed(random_seed)


data_dir = args.dataset


train_data = food101(data_dir, 'train', train_transforms)
valid_data = food101(data_dir, 'val', valid_transforms)


batch_size = args.batchsize
num_epochs = args.epoch

train_loader = data.DataLoader(train_data, batch_size, shuffle = True, num_workers = args.workers, pin_memory = True)
valid_loader = data.DataLoader(valid_data, batch_size*2, num_workers = args.workers, pin_memory = True)


if(args.arch == 'resnet34'):
	model = resnet.resnet34(101, 3, pretrained=args.pretrained)
elif(args.arch == 'resnet101'):
	model = resnet.resnet101(101, 3, pretrained=args.pretrained)
else:
	model = resnet.resnet50(101, 3, pretrained=args.pretrained)

model = model.cuda()


# max_lr = 0.1

optimizer = torch.optim.Adam(
	[
		{"params": model.fc_params(), "lr": args.lr},
		{'params': model.backbone_params()}
	],
	1e-4, weight_decay = args.weight_decay)
# optimizer = torch.optim.SGD(
# 	[
# 		{"params": model.fc_params(), "lr": 0.01},
# 		{'params': model.backbone_params()}
# 	],
# 	1e-4, momentum=0.9)


sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 16, 24], gamma=0.2)
# optimizer = torch.optim.Adam(model.parameters(), 0.001)
# criterion = nn.functional.cross_entropy().cuda()

best_acc = 0.0
start_epoch = 0


save_checkpoint = args.save_checkpoint
checkpoint_path = args.checkpoint_path
print_freq = args.print_freq


if(os.path.exists(checkpoint_path)):
	print("Saved model exists")
	checkpoint = torch.load(checkpoint_path)
	loss = checkpoint['loss']
	st_ep = checkpoint['epoch'] + 1
	best_acc = checkpoint['top1_accuracy']
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	sched.load_state_dict(checkpoint['scheduler_state_dict'])

	print("Top1 accuracy of saved model: %.2f%%" % best_acc)
	print("Top5 accuracy os saved model: %.2f%%" % checkpoint['top5_accuracy'] )

if(args.evaluate == True):

	with torch.no_grad():
		
		top1_acc = AverageMeter()
		top5_acc = AverageMeter()
		
		model.eval()

		for batch in valid_loader:
			images, labels = batch
			images = images.cuda()
			labels = labels.cuda()
			out = model(images)
			loss = nn.functional.cross_entropy(out, labels)
			prec = accuracy(out.detach(), labels, topk=(1,5))
# 			_, preds = torch.max(out, dim = 1)
# 			current_acc = torch.tensor(torch.sum(preds == labels).item()/len(preds)).item()
			top1_acc.update(prec[0])
			top5_acc.update(prec[1])
		
		print("Accuracy on test data: top1: %.2f%%   top5: %.2f%%" % (top1_acc.avg*100.0, top5_acc.avg*100.0))

	sys.exit()


fp = open("food101_training.txt", "w")


for epoch in range(start_epoch, num_epochs): 

	print("Epoch %d: \n" % epoch, flush=True)
	train_loss = training(train_loader, model, optimizer, print_freq)

	# print("Current learning rate: ", sched.get_last_lr(), flush=True)		

	sched.step()


	with torch.no_grad():

		model.eval()

		top1_acc = AverageMeter()
		top5_acc = AverageMeter()
		
		losses = AverageMeter()
		
		total_loss = 0.0

		for batch in valid_loader:
			images, labels = batch
			images = images.cuda()
			labels = labels.cuda()
			out = model(images)
			loss = nn.functional.cross_entropy(out, labels)
			
			prec = accuracy(out.detach(), labels, topk=(1,5))
# 			_, preds = torch.max(out, dim = 1)
# 			current_acc = torch.tensor(torch.sum(preds == labels).item()/len(preds)).item()
			
			losses.update(loss.item())
			top1_acc.update(prec[0])
			top5_acc.update(prec[1])


		print("Accuracy on test data: top1: %.2f%%, top5: %.2f%%, loss: %.2f" % ( (top1_acc.avg*100.0), (top5_acc.avg*100.0), losses.avg), flush=True)
		fp.write("%d %.2f %.2f %.4f %.4f\n" %(epoch, train_loss, losses.avg, top1_acc.avg, top5_acc.avg) )

		if(top1_acc.avg > best_acc and (save_checkpoint == True or epoch == num_epochs - 1)):
			best_acc = top1_acc.avg
			torch.save({
				'epoch': epoch,
				'top1_accuracy': (top1_acc.avg*100.0),
				'top5_accuracy': (top5_acc.avg*100.0),
				'loss': loss.item(),  
				'model_state_dict': model.state_dict(), 
				'optimizer_state_dict': optimizer.state_dict(), 
				'scheduler_state_dict': sched.state_dict()
				}, checkpoint_path)



fp.close()