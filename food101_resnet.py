import os
import torch 
import torch.nn as nn
import numpy as np 
import torchvision
import torch.utils.data as data
from torchvision import datasets, transforms
import resnet
from PIL import Image


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



def training(train_loader, model, optimizer):

	model.train()
	num = 0

	for batch in train_loader:
		optimizer.zero_grad()
		images, labels = batch
		images = images.cuda()
		labels = labels.cuda()
		out = model(images)
		loss = nn.functional.cross_entropy(out, labels)

		loss.backward()
		optimizer.step()

		# sched.step()
		if(num % 50 == 0):
			print(num, loss.item(), flush=True)
		num = num + 1



random_seed = 42
torch.manual_seed(random_seed)

data_dir = "./food-101"

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


# image_datasets = datasets.ImageFolder(data_dir, transform = train_transforms)

# val_size = int(len(image_datasets) * 0.1)
# train_size = len(image_datasets) - val_size

# train_data, valid_data = data.random_split(image_datasets, [train_size, val_size])

train_data = food101(data_dir, 'train', train_transforms)
valid_data = food101(data_dir, 'val', valid_transforms)


batch_size = 128
num_epochs = 50

train_loader = data.DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
valid_loader = data.DataLoader(valid_data, batch_size*2, num_workers = 4, pin_memory = True)


model = resnet.resnet50(101, 3, pretrained=True)
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
st_ep = 0


saved = "./best_checkpoint.pth"


if(os.path.exists(saved)):
	print("Saved model exists")
	checkpoint = torch.load(saved)
	loss = checkpoint['loss']
	st_ep = checkpoint['epoch'] + 1
	best_acc = checkpoint['accuracy']
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	sched.load_state_dict(checkpoint['scheduler_state_dict'])

	print("Best accuracy is: %f" % best_acc)



for epoch in range(st_ep, num_epochs): 

	print("Epoch %d: \n" % epoch, flush=True)
	training(train_loader, model, optimizer)

	print("Current learning rate: ", sched.get_last_lr(), flush=True)		

	sched.step()

	model.eval()


	with torch.no_grad():
		num = 0
		acc = 0.0
		total_loss = 0.0
		for batch in valid_loader:
			images, labels = batch
			images = images.cuda()
			labels = labels.cuda()
			out = model(images)
			loss = nn.functional.cross_entropy(out, labels)
			_, preds = torch.max(out, dim = 1)
			acc = acc + torch.tensor(torch.sum(preds == labels).item()/len(preds)).item()
			total_loss = total_loss + loss
			num = num + 1


		acc = acc / num
		print("Accuracy on test data: %f, loss: %f" % (acc, total_loss), flush=True)

		if(acc > best_acc):
			best_acc = acc
			torch.save({
				'epoch': epoch,
				'accuracy': acc, 
				'loss': loss.item(),  
				'model_state_dict': model.state_dict(), 
				'optimizer_state_dict': optimizer.state_dict(), 
				'scheduler_state_dict': sched.state_dict()
				}, saved)















