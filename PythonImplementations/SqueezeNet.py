import torch
from torch import nn, Tensor
from torch.functional import F
import torch.nn.init as init
from utils import Fire


class SqueezeNet(nn.Module):
	def __init__(self, class_count: int):
		super(SqueezeNet, self).__init__()
		self._features = nn.Sequential(
			nn.Conv2d(
				in_channels=3,
				out_channels=96,
				kernel_size=(7, 7),
				stride=(2, 2)
			),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True),
			Fire(
				in_channels=96,
				squeeze_filter_count=16,
				expand_1x1_filter_count=64,
				expand_3x3_filter_count=64
			),
			Fire(
				in_channels=128,
				squeeze_filter_count=16,
				expand_1x1_filter_count=64,
				expand_3x3_filter_count=64
			),
			Fire(
				in_channels=128,
				squeeze_filter_count=32,
				expand_1x1_filter_count=128,
				expand_3x3_filter_count=128
			),
			nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True),
			Fire(
				in_channels=256,
				squeeze_filter_count=32,
				expand_1x1_filter_count=128,
				expand_3x3_filter_count=128
			),
			Fire(
				in_channels=256,
				squeeze_filter_count=48,
				expand_1x1_filter_count=192,
				expand_3x3_filter_count=192
			),
			Fire(
				in_channels=384,
				squeeze_filter_count=48,
				expand_1x1_filter_count=192,
				expand_3x3_filter_count=192
			),
			Fire(
				in_channels=384,
				squeeze_filter_count=64,
				expand_1x1_filter_count=256,
				expand_3x3_filter_count=256
			),
			nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True),
			Fire(
				in_channels=512,
				squeeze_filter_count=64,
				expand_1x1_filter_count=256,
				expand_3x3_filter_count=256
			)
		)

		self._last_conv2d = nn.Conv2d(
			in_channels=512,
			out_channels=class_count,
			kernel_size=(1, 1)
		)

		self._classifier = nn.Sequential(
			nn.Dropout(p=0.5),
			self._last_conv2d,
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d(output_size=(1, 1)),
			nn.Softmax(dim=1)
		)

		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				if module is self._last_conv2d:
					init.normal_(tensor=module.weight, mean=0.0, std=0.01)
				else:
					init.kaiming_uniform_(tensor=module.weight)
				if module.bias is not None:
					init.constant_(tensor=module.bias, val=0.0)

	def forward(self, x: Tensor) -> Tensor:
		x = self._features(x)
		for i, image in enumerate(x):
			if i == 0:
				print(f"=========================={i} layer==========================")
				for j, channel in enumerate(image):
					print(f"{i:03} {j:03}: {channel[0][0]:.2f}", end=" ")
					if j != 0 and j % 10 == 0:
						print()
		x = self._classifier(x)

		return torch.flatten(input=x, start_dim=1)


class SqueezeResNet(nn.Module):
	def __init__(self, class_count: int):
		super(SqueezeResNet, self).__init__()
		self._conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=3,
				out_channels=96,
				kernel_size=(7, 7),
				stride=(2, 2)
			),
			nn.ReLU(inplace=True)
		)
		self._maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True)
		self._fire2 = Fire(
			in_channels=96,
			squeeze_filter_count=16,
			expand_1x1_filter_count=64,
			expand_3x3_filter_count=64
		)
		self._fire3 = Fire(
			in_channels=128,
			squeeze_filter_count=16,
			expand_1x1_filter_count=64,
			expand_3x3_filter_count=64
		)
		self._fire4 = Fire(
			in_channels=128,
			squeeze_filter_count=32,
			expand_1x1_filter_count=128,
			expand_3x3_filter_count=128
		)
		self._maxpool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True)
		self._fire5 = Fire(
			in_channels=256,
			squeeze_filter_count=32,
			expand_1x1_filter_count=128,
			expand_3x3_filter_count=128
		)
		self._fire6 = Fire(
			in_channels=256,
			squeeze_filter_count=48,
			expand_1x1_filter_count=192,
			expand_3x3_filter_count=192
		)
		self._fire7 = Fire(
			in_channels=384,
			squeeze_filter_count=48,
			expand_1x1_filter_count=192,
			expand_3x3_filter_count=192
		)
		self._fire8 = Fire(
			in_channels=384,
			squeeze_filter_count=64,
			expand_1x1_filter_count=256,
			expand_3x3_filter_count=256
		)
		self._maxpool8 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True)
		self._fire9 = Fire(
			in_channels=512,
			squeeze_filter_count=64,
			expand_1x1_filter_count=256,
			expand_3x3_filter_count=256
		)

		self._conv10 = nn.Conv2d(
			in_channels=512,
			out_channels=class_count,
			kernel_size=(1, 1)
		)

		self._classifier = nn.Sequential(
			nn.Dropout(p=0.5),
			self._conv10,
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d(output_size=(1, 1))
		)

		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				if module is self._conv10:
					init.normal_(tensor=module.weight, mean=0.0, std=0.01)
				else:
					init.kaiming_uniform_(tensor=module.weight)
				if module.bias is not None:
					init.constant_(tensor=module.bias, val=0.0)

	def forward(self, x: Tensor) -> Tensor:
		x = self._fire2(self._maxpool1(self._conv1(x)))
		res_x = x
		x = res_x + self._fire3(x)
		x = self._maxpool4(self._fire4(x))
		res_x = x
		x = res_x + self._fire5(x)
		x = self._fire6(x)
		res_x = x
		x = res_x + self._fire7(x)
		x = self._maxpool8(self._fire8(x))
		res_x = x
		x = res_x + self._fire9(x)
		x = self._classifier(x)

		return torch.flatten(input=x, start_dim=1)


class SqueezeComplexResNet(nn.Module):
	def __init__(self, class_count: int):
		super(SqueezeComplexResNet, self).__init__()
		self._conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=3,
				out_channels=96,
				kernel_size=(7, 7),
				stride=(2, 2)
			),
			nn.ReLU(inplace=True)
		)
		self._maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True)
		self._res_conv2d_1 = nn.Sequential(
			nn.Conv2d(
				in_channels=96,
				out_channels=128,
				kernel_size=(1, 1)
			),
			nn.ReLU(
				inplace=True
			)
		)
		self._fire2 = Fire(
			in_channels=96,
			squeeze_filter_count=16,
			expand_1x1_filter_count=64,
			expand_3x3_filter_count=64
		)
		self._fire3 = Fire(
			in_channels=128,
			squeeze_filter_count=16,
			expand_1x1_filter_count=64,
			expand_3x3_filter_count=64
		)
		self._res_conv2d_3 = nn.Sequential(
			nn.Conv2d(
				in_channels=128,
				out_channels=256,
				kernel_size=(1, 1)
			),
			nn.ReLU(
				inplace=True
			)
		)
		self._fire4 = Fire(
			in_channels=128,
			squeeze_filter_count=32,
			expand_1x1_filter_count=128,
			expand_3x3_filter_count=128
		)
		self._maxpool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True)
		self._fire5 = Fire(
			in_channels=256,
			squeeze_filter_count=32,
			expand_1x1_filter_count=128,
			expand_3x3_filter_count=128
		)
		self._res_conv2d_5 = nn.Sequential(
			nn.Conv2d(
				in_channels=256,
				out_channels=384,
				kernel_size=(1, 1)
			),
			nn.ReLU(
				inplace=True
			)
		)
		self._fire6 = Fire(
			in_channels=256,
			squeeze_filter_count=48,
			expand_1x1_filter_count=192,
			expand_3x3_filter_count=192
		)
		self._fire7 = Fire(
			in_channels=384,
			squeeze_filter_count=48,
			expand_1x1_filter_count=192,
			expand_3x3_filter_count=192
		)
		self._res_conv2d_7 = nn.Sequential(
			nn.Conv2d(
				in_channels=384,
				out_channels=512,
				kernel_size=(1, 1)
			),
			nn.ReLU(
				inplace=True
			)
		)
		self._fire8 = Fire(
			in_channels=384,
			squeeze_filter_count=64,
			expand_1x1_filter_count=256,
			expand_3x3_filter_count=256
		)
		self._maxpool8 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True)
		self._fire9 = Fire(
			in_channels=512,
			squeeze_filter_count=64,
			expand_1x1_filter_count=256,
			expand_3x3_filter_count=256
		)

		self._conv10 = nn.Conv2d(
			in_channels=512,
			out_channels=class_count,
			kernel_size=(1, 1)
		)

		self._classifier = nn.Sequential(
			nn.Dropout(p=0.5),
			self._conv10,
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d(output_size=(1, 1))
		)

		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				if module is self._conv10:
					init.normal_(tensor=module.weight, mean=0.0, std=0.01)
				else:
					init.kaiming_uniform_(tensor=module.weight)
				if module.bias is not None:
					init.constant_(tensor=module.bias, val=0.0)

	def forward(self, x: Tensor) -> Tensor:
		x = self._maxpool1(self._conv1(x))
		x = self._res_conv2d_1(x) + self._fire2(x)
		x = x + self._fire3(x)
		x = self._res_conv2d_3(x) + self._fire4(x)
		x = self._maxpool4(x)
		x = x + self._fire5(x)
		x = self._res_conv2d_5(x) + self._fire6(x)
		x = x + self._fire7(x)
		x = self._res_conv2d_7(x) + self._fire8(x)
		x = self._maxpool8(x)
		x = x + self._fire9(x)
		x = self._classifier(x)

		return torch.flatten(input=x, start_dim=1)