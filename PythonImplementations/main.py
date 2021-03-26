from enum import Enum
import os
import os.path
import sys
import time
import urllib

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn, Tensor
from torch.functional import F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from DenseNet import DenseNet
from SqueezeNet import SqueezeNet, SqueezeResNet, SqueezeComplexResNet

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


class Dataset(Enum):
	IMAGENET = 0
	CIFAR10 = 1
	CIFAR100 = 2


def main():
	if sys.argv.__contains__("cifar100"):
		dataset = Dataset.CIFAR100
		class_count = 100
	elif sys.argv.__contains__("cifar10"):
		dataset = Dataset.CIFAR10
		class_count = 10
	elif sys.argv.__contains__("imagenet"):
		dataset = Dataset.IMAGENET
		class_count = 1000
	else:
		dataset = Dataset.CIFAR100
		class_count = 100

	if sys.argv.__contains__("densenet"):
		model = DenseNet(
			layer_count=40,
			growth_rate=12,
			compression_factor=0.5,
			class_count=class_count,
			dropout_rate=0.2,
			device=device
		).to(device)
		loss_function = nn.CrossEntropyLoss()
		optimizer = torch.optim.SGD(
			params=model.parameters(),
			lr=0.1,
			momentum=0.9,
			weight_decay=0.0001,
			nesterov=True
		)
		epochs = 300
		scheduler = torch.optim.lr_scheduler.MultiStepLR(
			optimizer=optimizer,
			milestones=[
				int(0.5 * epochs),
				int(0.75 * epochs)
			],
			gamma=0.1
		)
		model_file_name = "densenet_model"
		valid_set_count = 5000
		need_to_scale = False
	elif sys.argv.__contains__("densenet121"):
		model = DenseNet(
			layer_count=121,
			growth_rate=32,
			compression_factor=0.5,
			class_count=class_count,
			dropout_rate=0.2,
			layers_count_by_block=[6, 12, 24, 16],
			device=device
		).to(device)
		loss_function = nn.CrossEntropyLoss()
		optimizer = torch.optim.SGD(
			params=model.parameters(),
			lr=0.1,
			momentum=0.9,
			weight_decay=0.0001,
			nesterov=True
		)
		epochs = 300
		scheduler = torch.optim.lr_scheduler.MultiStepLR(
			optimizer=optimizer,
			milestones=[
				int(0.5 * epochs),
				int(0.75 * epochs)
			],
			gamma=0.1
		)
		model_file_name = "densenet121_model"
		valid_set_count = 5000
		need_to_scale = True
	elif sys.argv.__contains__("densenet169"):
		model = DenseNet(
			layer_count=169,
			growth_rate=32,
			compression_factor=0.5,
			class_count=class_count,
			dropout_rate=0.2,
			layers_count_by_block=[6, 12, 32, 32],
			device=device
		).to(device)
		loss_function = nn.CrossEntropyLoss()
		optimizer = torch.optim.SGD(
			params=model.parameters(),
			lr=0.1,
			momentum=0.9,
			weight_decay=0.0001,
			nesterov=True
		)
		epochs = 300
		scheduler = torch.optim.lr_scheduler.MultiStepLR(
			optimizer=optimizer,
			milestones=[
				int(0.5 * epochs),
				int(0.75 * epochs)
			],
			gamma=0.1
		)
		model_file_name = "densenet169_model"
		valid_set_count = 5000
		need_to_scale = True
	elif sys.argv.__contains__("densenet201"):
		model = DenseNet(
			layer_count=201,
			growth_rate=32,
			compression_factor=0.5,
			class_count=class_count,
			dropout_rate=0.2,
			layers_count_by_block=[6, 12, 48, 32],
			device=device
		).to(device)
		loss_function = nn.CrossEntropyLoss()
		optimizer = torch.optim.SGD(
			params=model.parameters(),
			lr=0.1,
			momentum=0.9,
			weight_decay=0.0001,
			nesterov=True
		)
		epochs = 300
		scheduler = torch.optim.lr_scheduler.MultiStepLR(
			optimizer=optimizer,
			milestones=[
				int(0.5 * epochs),
				int(0.75 * epochs)
			],
			gamma=0.1
		)
		model_file_name = "densenet169_model"
		valid_set_count = 5000
		need_to_scale = True
	elif sys.argv.__contains__("densenet161"):
		model = DenseNet(
			layer_count=161,
			growth_rate=48,
			compression_factor=0.5,
			class_count=class_count,
			dropout_rate=0.2,
			layers_count_by_block=[6, 12, 36, 24],
			device=device
		).to(device)
		loss_function = nn.CrossEntropyLoss()
		optimizer = torch.optim.SGD(
			params=model.parameters(),
			lr=0.1,
			momentum=0.9,
			weight_decay=0.0001,
			nesterov=True
		)
		epochs = 300
		scheduler = torch.optim.lr_scheduler.MultiStepLR(
			optimizer=optimizer,
			milestones=[
				int(0.5 * epochs),
				int(0.75 * epochs)
			],
			gamma=0.1
		)
		model_file_name = "densenet169_model"
		valid_set_count = 5000
		need_to_scale = True
	elif sys.argv.__contains__("squeezenet"):
		model = SqueezeNet(
			class_count=class_count
		).to(device)
		model_file_name = "squeezenet_model"
		loss_function = nn.CrossEntropyLoss()
		optimizer = torch.optim.SGD(
			params=model.parameters(),
			lr=0.04
		)
		epochs = 300
		scheduler = torch.optim.lr_scheduler.StepLR(
			optimizer=optimizer,
			step_size=1,
			gamma=64.0 / 256.0
		)
		valid_set_count = 5000
		need_to_scale = False
	elif sys.argv.__contains__("squeezeresnet"):
		model = SqueezeResNet(
			class_count=class_count
		).to(device)
		model_file_name = "squeezeresnet_model"
		loss_function = nn.CrossEntropyLoss()
		optimizer = torch.optim.SGD(
			params=model.parameters(),
			lr=0.04
		)
		epochs = 300
		scheduler = torch.optim.lr_scheduler.StepLR(
			optimizer=optimizer,
			step_size=1,
			gamma=64.0 / 256.0
		)
		valid_set_count = 5000
		need_to_scale = True
	elif sys.argv.__contains__("squeezecomplexresnet"):
		model = SqueezeComplexResNet(
			class_count=class_count
		).to(device)
		model_file_name = "squeezecomplexresnet_model"
		loss_function = nn.CrossEntropyLoss()
		optimizer = torch.optim.SGD(
			params=model.parameters(),
			lr=0.04
		)
		epochs = 300
		scheduler = torch.optim.lr_scheduler.StepLR(
			optimizer=optimizer,
			step_size=1,
			gamma=64.0 / 256.0
		)
		valid_set_count = 5000
		need_to_scale = True
	else:
		model = DenseNet(
			layer_count=40,
			growth_rate=12,
			compression_factor=0.5,
			class_count=class_count,
			dropout_rate=0.2,
			device=device
		).to(device)
		loss_function = nn.CrossEntropyLoss()
		optimizer = torch.optim.SGD(
			params=model.parameters(),
			lr=0.1,
			momentum=0.9,
			weight_decay=0.0001,
			nesterov=True
		)
		epochs = 300
		scheduler = torch.optim.lr_scheduler.MultiStepLR(
			optimizer=optimizer,
			milestones=[
				int(0.5 * epochs),
				int(0.75 * epochs)
			],
			gamma=0.1
		)
		model_file_name = "densenet_model"
		need_to_scale = False
		valid_set_count = 5000

	train_dataloader, test_dataloader, valid_dataloader = load_data(
		dataset=dataset,
		valid_set_count=valid_set_count,
		need_to_scale=need_to_scale
	)

	print("Current model name: " + model_file_name)
	max_epoch = 0
	for file in os.listdir():
		if file.startswith(f"{model_file_name}"):
			split_file_name = file.split('_')
			if split_file_name[0] + '_' + split_file_name[1] == model_file_name:
				max_epoch = max(max_epoch, int(split_file_name[-1][:-4]))

	if os.path.isfile(f"{model_file_name}_{dataset}_{device}_{max_epoch}.pth"):
		load_model(model, f"{model_file_name}_{dataset}_{device}_{max_epoch}.pth")
	else:
		max_epoch = 0

	train_flag = False
	test_flag = False
	valid_flag = False
	topk_flag = False
	if sys.argv.__contains__("-train"):
		train_flag = True
	if sys.argv.__contains__("-test"):
		test_flag = True
	if sys.argv.__contains__("-valid"):
		valid_flag = True
	if sys.argv.__contains__("-topk"):
		topk_flag = True

	last_epoch = 0
	if train_flag or test_flag or valid_flag:
		try:
			for t in range(max_epoch, epochs):
				print(f"Epoch {t + 1}\n-------------------------------")
				if train_flag:
					train(train_dataloader, model, loss_function, optimizer)
				scheduler.step()
				if test_flag:
					test(test_dataloader, model, loss_function, False)
				if valid_flag and valid_dataloader:
					test(valid_dataloader, model, loss_function, True)

				last_epoch = t + 1
			print("Done!")
		except Exception:
			print(Exception)
			save_model(model, f"{model_file_name}_{dataset}_{device}_{last_epoch}.pth")
		except KeyboardInterrupt:
			save_model(model, f"{model_file_name}_{dataset}_{device}_{last_epoch}.pth")
			return

	if topk_flag and valid_dataloader:
		topk_accuracy(dataloader=valid_dataloader, model=model)

	save_model(model, f"{model_file_name}_{dataset}_{device}_{last_epoch}.pth")


def download_data(dataset: Dataset, valid_set_count: int, need_to_scale: bool):
	# Download training data from open datasets
	if dataset == Dataset.IMAGENET:
		training_data = datasets.CIFAR10(
			root="data",
			train=True,
			download=True,
			transform=ToTensor()
		)
		test_data = datasets.CIFAR10(
			root="data",
			train=False,
			download=True,
			transform=ToTensor()
		)
	elif dataset == Dataset.CIFAR10:
		mean = [0.49139968, 0.48215841, 0.44653091]
		std = [0.24703223, 0.24348513, 0.26158784]
		if need_to_scale:
			train_transforms = transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std),
			])
			test_transforms = transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std),
			])
		else:
			train_transforms = transforms.Compose([
				transforms.RandomCrop(32, padding=4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std),
			])
			test_transforms = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std),
			])
		training_data = datasets.CIFAR10(
			root="data",
			train=True,
			download=True,
			transform=train_transforms
		)
		test_data = datasets.CIFAR10(
			root="data",
			train=False,
			download=True,
			transform=test_transforms
		)

		if valid_set_count > 0:
			valid_data = datasets.CIFAR10(
				root="data",
				train=True,
				transform=test_transforms
			)
			indices = torch.randperm(len(training_data))
			train_indices = indices[:len(indices) - valid_set_count]
			valid_indices = indices[len(indices) - valid_set_count:]
			training_data = torch.utils.data.Subset(training_data, train_indices)
			valid_data = torch.utils.data.Subset(valid_data, valid_indices)
		else:
			valid_data = None
	elif dataset == Dataset.CIFAR100:
		mean = [n / 255.0 for n in [129.3, 124.1, 112.4]]
		std = [n / 255.0 for n in [68.2,  65.4,  70.4]]
		if need_to_scale:
			train_transforms = transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std),
			])
			test_transforms = transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std),
			])
		else:
			train_transforms = transforms.Compose([
				transforms.RandomCrop(32, padding=4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std),
			])
			test_transforms = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std),
			])
		training_data = datasets.CIFAR100(
			root="data",
			train=True,
			download=True,
			transform=train_transforms
		)
		test_data = datasets.CIFAR100(
			root="data",
			train=False,
			download=True,
			transform=test_transforms
		)

		if valid_set_count > 0:
			valid_data = datasets.CIFAR100(
				root="data",
				train=True,
				transform=test_transforms
			)
			indices = torch.randperm(len(training_data))
			train_indices = indices[:len(indices) - valid_set_count]
			valid_indices = indices[len(indices) - valid_set_count:]
			training_data = torch.utils.data.Subset(training_data, train_indices)
			valid_data = torch.utils.data.Subset(valid_data, valid_indices)
		else:
			valid_data = None

	return training_data, test_data, valid_data


def load_data(dataset: Dataset, batch_size: int = 64, valid_set_count: int = 5000, need_to_scale: bool = False):
	training_data, test_data, valid_data = download_data(dataset, valid_set_count, need_to_scale)
	print("Using dataset: " + str(dataset))

	# Create data loaders
	train_dataloader = DataLoader(training_data, batch_size=batch_size)
	test_dataloader = DataLoader(test_data, batch_size=batch_size)
	if valid_data is not None:
		valid_dataloader = DataLoader(valid_data, batch_size=batch_size)
	else:
		valid_dataloader = None

	for x, y in test_dataloader:
		print("Shape of x [N, C, H, W]: ", x.shape)
		print("Shape of y: ", y.shape, y.dtype)
		break

	return train_dataloader, test_dataloader, valid_dataloader


def load_model(model: nn.Module, model_file_name: str):
	assert isinstance(model, nn.Module)
	assert isinstance(model_file_name, str)

	model.load_state_dict(torch.load(f"pretrained/{model_file_name}"))

	print("Loaded PyTorch Model State {}".format(model_file_name))


def predict(model: nn.Module, test_data: torch.utils.data.Dataset):
	classes = []
	if isinstance(test_data, datasets.FashionMNIST):
		classes = [
			"T-shirt/top",
			"Trouser",
			"Pullover",
			"Dress",
			"Coat",
			"Sandal",
			"Shirt",
			"Sneaker",
			"Bag",
			"Ankle boot"
		]

	model.eval()
	x, y = test_data[0][0], test_data[0][1]
	with torch.no_grad():
		pred = model(x)
		predicted, actual = classes[pred[0].argmax(0)], classes[y]
		print(f"Predicted: \"{predicted}\", Actual: \n{actual}\n")


def predict_image(filename: str, model: nn.Module):
	url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
	try:
		urllib.URLopener().retrieve(url, filename)
	except:
		urllib.request.urlretrieve(url, filename)

	input_image: Image.Image = Image.open(filename)
	preprocess = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)
	])
	input_tensor: Tensor = preprocess(input_image)
	input_batch: Tensor = input_tensor.unsqueeze(0)
	input_batch.to(device)
	model.to(device)

	with torch.no_grad():
		output: Tensor = model(input_batch)

	print(output[0])

	probabilities = F.softmax(output[0], dim=0)
	print(probabilities)

	with open("data/imagenet_classes.txt", 'r') as imagenet_classes:
		categories = [s.strip() for s in imagenet_classes.readlines()]

	top5_prob, top5_catid = torch.topk(probabilities, 5)
	for i in range(top5_prob.size(0)):
		print(categories[top5_catid[i]], top5_prob[i].item())


def save_model(model: nn.Module, filename: str):
	assert isinstance(model, nn.Module)

	torch.save(model.state_dict(), f"pretrained/{filename}")
	print("Saved PyTorch Model State to " + filename)


def train(dataloader: DataLoader, model: nn.Module, loss_function: nn.Module, optimizer: torch.optim.Optimizer):
	assert isinstance(dataloader, DataLoader)
	assert isinstance(model, nn.Module)
	assert isinstance(loss_function, nn.Module)
	assert isinstance(optimizer, torch.optim.Optimizer)

	tic = time.perf_counter()
	model.train()
	size = len(dataloader.dataset)
	for batch, (x, y) in enumerate(dataloader):
		x, y = x.to(device), y.to(device)


		# Compute prediction error
		pred: Tensor = model(x)
		loss = loss_function(pred, y)


		# Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 100 == 0:
			loss, current = loss.item(), batch * len(x)
			print(f"loss: {loss:>7f} [{current:5>d}/{size:>5d}]")
	toc = time.perf_counter()
	print(f"Train took {toc - tic}")


def test(dataloader: DataLoader, model: nn.Module, loss_function, is_validation: bool = False):
	assert isinstance(dataloader, DataLoader)
	assert isinstance(model, nn.Module)
	assert isinstance(loss_function, nn.Module)
	assert isinstance(is_validation, bool)

	size = len(dataloader.dataset)
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for x, y in dataloader:
			x, y = x.to(device), y.to(device)
			pred = model(x)
			test_loss += loss_function(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
	test_loss /= size
	correct /= size
	print(f"{'Validation' if is_validation else 'Test'} Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def topk_accuracy(dataloader: DataLoader, model: nn.Module):
	top_1 = 0
	top_5 = 0

	with torch.no_grad():
		for i, (x, y) in enumerate(dataloader):
			x, y = x.to(device), y.to(device)
			pred: Tensor = model(x)
			_, pred = torch.topk(input=pred, k=5, dim=1)
			correct = pred.eq(y.view(-1, 1).expand_as(pred)).cpu().data.numpy()
			top_1 += correct[:, 0].sum()
			top_5 += correct.sum()
			print("{} top1: {} top5: {}".format(i, top_1, top_5))
		print("Top1 accuracy: ", top_1 / len(dataloader.dataset))
		print("Top5 accuracy: ", top_5 / len(dataloader.dataset))


if __name__ == "__main__":
	print("ARGV:", sys.argv)
	main()
