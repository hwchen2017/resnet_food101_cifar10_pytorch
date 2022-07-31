import os
import torch 
import torch.nn as nn
import numpy as np 
import torchvision
import torch.utils.data as data
from torchvision import datasets, transforms
import resnet
from PIL import Image
from template import AverageMeter


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



parser = argparse.ArgumentParser(description = 'Pytorch Resnet Training for Food101')
parser.add_argument('--epoch', help='Number of training epoches', type=int, default=30)
parser.add_argument('--arch', help='Architecture of Resnet Model: resnet34, resnet50, resnet101', type=str, default='resnet50')
parser.add_argument('--batchsize', help = 'Size of training batch', type = int, default = 64)
parser.add_argument('--lr', help = 'Laraning rate', type = float, default = 0.01)
parser.add_argument('--weight_decay', help = "Weight decay", type = float, default = 1e-4)
parser.add_argument('--workers', help = 'Number of worker', type = int, default = 4)
parser.add_argument('--dataset', help = 'The path of training data', type = str, default = './food-101')
parser.add_argument('--pretrained', help = 'Using imagenet pretrained model', type = bool, default = True)
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


model = resnet.resnet50(101, 3, pretrained=args.pretrained)
model = model.cuda()


# max_lr = 0.1

optimizer = torch.optim.Adam(
	[
		{"params": model.fc_params(), "lr": 0.01},
		{'params': model.backbone_params()}
	],
	1e-4, weight_decay = 1e-4)
# optimizer = torch.optim.SGD(
# 	[
# 		{"params": model.fc_params(), "lr": 0.01},
# 		{'params': model.backbone_params()}
# 	],
# 	1e-4, momentum=0.9)





sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.2)
# optimizer = torch.optim.Adam(model.parameters(), 0.001)
# criterion = nn.functional.cross_entropy().cuda()

best_acc = 0.0
start_epoch = 0


save_checkpoint = args.saved_checkpoint
checkpoint_path = args.checkpoint_path


if(os.path.exists(checkpoint_path)):
	print("Saved model exists")
	checkpoint = torch.load(checkpoint_path)
	loss = checkpoint['loss']
	st_ep = checkpoint['epoch'] + 1
	best_acc = checkpoint['accuracy']
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	sched.load_state_dict(checkpoint['scheduler_state_dict'])

	print("Accuracy of saved model is:  %.2f%%" % best_acc)


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


for epoch in range(st_ep, num_epochs): 

	print("Epoch %d: \n" % epoch, flush=True)
	training(train_loader, model, optimizer)

	# print("Current learning rate: ", sched.get_last_lr(), flush=True)		

	sched.step()


	with torch.no_grad():

		model.eval()

		acc = AverageMeter()
		losses = AverageMeter()
		
		total_loss = 0.0

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
			# total_loss = total_loss + loss
			# num = num + 1


		# acc = acc / num
		print("Accuracy on test data: %.2f%%, loss: %.2f" % ((acc.avg*100.0), losses.avg), flush=True)

		if(acc.avg > best_acc and (save_checkpoint == True or epoch == num_epochs - 1)):
			best_acc = acc.avg
			torch.save({
				'epoch': epoch,
				'accuracy': (acc.avg*100.0), 
				'loss': loss.item(),  
				'model_state_dict': model.state_dict(), 
				'optimizer_state_dict': optimizer.state_dict(), 
				'scheduler_state_dict': sched.state_dict()
				}, checkpoint_path)















