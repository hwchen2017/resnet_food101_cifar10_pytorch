import torch.nn as nn
import math 
import torch.utils.model_zoo as model_zoo

model_urls = {
	'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    # 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet50': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1h2_176-001a1197.pth', 
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride = 1):
	return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)


class BasicBlock(nn.Module):

	expansion = 1

	def __init__(self, inplanes, outplanes, stride = 1, downsample = None):
		super().__init__()
		self.conv1 = conv3x3(inplanes, outplanes, stride)
		self.bn1 = nn.BatchNorm2d(outplanes)
		self.relu = nn.ReLU(inplace = True)
		
		self.conv2 = conv3x3(outplanes, outplanes)
		self.bn2 = nn.BatchNorm2d(outplanes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):

		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if(self.downsample is not None):
			residual = self.downsample(x)

		out += residual

		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, outplanes, stride = 1, downsample = None):
		super().__init__()
		self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size = 1, bias = False)
		self.bn1 = nn.BatchNorm2d(outplanes)
		
		self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size = 3, stride = stride, padding = 1, bias = False)
		self.bn2 = nn.BatchNorm2d(outplanes)

		self.conv3 = nn.Conv2d(outplanes, outplanes*self.expansion, kernel_size = 1, bias = False)
		self.bn3 = nn.BatchNorm2d(outplanes*self.expansion)

		self.relu = nn.ReLU(inplace = True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x): 
		residual = x 

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if(self.downsample is not None):
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out



class ResNet(nn.Module):

	def __init__(self, block, layer, num_class = 1000, input_chanel = 3):
		super().__init__()
		self.inplanes = 64
		self.conv1 = nn.Conv2d(input_chanel, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace = True)
		self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

		self.layer1 = self.make_layer(block, 64, layer[0])
		self.layer2 = self.make_layer(block, 128, layer[1], stride = 2)
		self.layer3 = self.make_layer(block, 256, layer[2], stride = 2)
		self.layer4 = self.make_layer(block, 512, layer[3], stride = 2)

		self.avgpool = nn.AvgPool2d(7, stride = 2)

		self.dropout = nn.Dropout2d(p = 0.5, inplace = True)

		self.fc = nn.Linear(512*block.expansion, num_class)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2.0/n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def make_layer(self, block, planes, blocks, stride = 1):
		downsample = None

		if(stride !=1 or self.inplanes != planes * block.expansion):
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size = 1, stride = stride, bias = False), 
				nn.BatchNorm2d(planes*block.expansion))

		layers = []

		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))


		return nn.Sequential(*layers)

	def fc_params(self):
		params = []
		for name, param in self.named_parameters():
			if 'fc' in name:
				params.append(param)
		return params

	def backbone_params(self):
		params = []
		for name, param in self.named_parameters():
			if 'fc' not in name:
				params.append(param)
		return params

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = self.dropout(x)

		x = x.view(x.size(0), -1)

		x = self.fc(x)

		return x


def resnet34(num_class = 101, input_chanel = 3, pretrained=False):

	model = ResNet(BasicBlock, [3, 4, 6, 3], num_class, input_chanel)

	if(pretrained == True):
		state_dict = model_zoo.load_url(model_urls['resnet34'], map_location='cpu')
		state_dict.pop('fc.weight', None)
		state_dict.pop('fc.bias', None)

		missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
		print("Missing keys: ", missing_keys)
		print("Unexpected keys: ", unexpected_keys)

		print("Loaded imagenet pretained model")

	return model




def resnet50(num_class = 101, input_chanel = 3, pretrained=False):

	model = ResNet(Bottleneck, [3, 4, 6, 3], num_class, input_chanel)

	if(pretrained == True):
		state_dict = model_zoo.load_url(model_urls['resnet50'], map_location='cpu')
		state_dict.pop('fc.weight', None)
		state_dict.pop('fc.bias', None)

		missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
		print("Missing keys: ", missing_keys)
		print("Unexpected keys: ", unexpected_keys)

		print("Loaded imagenet pretained model")

	return model

def resnet101(num_class = 101, input_chanel = 3, pretrained=False):

	model = ResNet(Bottleneck, [3, 4, 23, 3], num_class, input_chanel)

	if(pretrained == True):
		state_dict = model_zoo.load_url(model_urls['resnet101'], map_location='cpu')
		state_dict.pop('fc.weight', None)
		state_dict.pop('fc.bias', None)

		missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
		print("Missing keys: ", missing_keys)
		print("Unexpected keys: ", unexpected_keys)

		print("Loaded imagenet pretained model")

	return model










