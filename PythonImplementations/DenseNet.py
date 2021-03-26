from collections import OrderedDict
from typing import List

import torch
from torch import nn, Tensor
from torch.functional import F
from utils import DenseNetConv2d
import cv2
import numpy as np


class DenseNet(nn.Module):
	class DenseBlock(nn.Module):
		def __init__(
				self,
				in_channels: int,
				growth_rate: int,
				layer_count: int,
				dropout_rate: float = 0.0,
				is_bottleneck: bool = False
		):
			assert isinstance(in_channels, int)
			assert isinstance(growth_rate, int)
			assert isinstance(layer_count, int)
			assert isinstance(dropout_rate, float)
			assert isinstance(is_bottleneck, bool)
			super(DenseNet.DenseBlock, self).__init__()

			for i in range(layer_count):
				self.add_module(
					f"dense_block_{i + 1}",
					DenseNetConv2d(
						in_channels=in_channels + i * growth_rate,
						growth_rate=growth_rate,
						kernel_size=(3, 3),
						padding=(1, 1),
						is_bottleneck=is_bottleneck,
						dropout_rate=dropout_rate
					)
				)

		def forward(self, x: Tensor) -> Tensor:
			assert isinstance(x, Tensor)
			# print("DenseNet::DenseBlock::forward")
			x_data = [x]
			for name, layer in self.named_children():
				x_data.append(
					layer(torch.cat(x_data, 1))
				)

			return torch.cat(x_data, 1)

	class TransitionLayer(nn.Sequential):
		def __init__(self, in_channels: int, compression_factor: float, dropout_rate: float = 0.0):
			assert isinstance(compression_factor, float) and 0 < compression_factor <= 1
			assert isinstance(in_channels, int)
			super(DenseNet.TransitionLayer, self).__init__()
			self.add_module(
				"conv",
				DenseNetConv2d(
					in_channels=in_channels,
					growth_rate=int(compression_factor * in_channels),
					kernel_size=(1, 1),
					dropout_rate=dropout_rate
				)
			)
			self.add_module(
				"pool",
				nn.MaxPool2d(kernel_size=(2, 2), stride=2)
			)

	def __init__(
			self,
			layer_count: int,
			growth_rate: int,
			compression_factor: float,
			class_count: int,
			dropout_rate: float = 0.0,
			layers_count_by_block: List[int] = None,
			device: str = "cpu"
	):
		assert isinstance(layer_count, int)
		assert isinstance(growth_rate, int)
		assert isinstance(class_count, int)
		assert isinstance(compression_factor, float) and 0.0 < compression_factor <= 1.0
		assert isinstance(dropout_rate, float) and 0.0 <= dropout_rate <= 1.0
		assert device == "cpu" or device == "cuda"
		super(DenseNet, self).__init__()
		self.device = device

		self._layer_count = int((layer_count - 4) / 3)

		if layers_count_by_block is not None:
			self._layer_count /= 2
			self.features = nn.Sequential(OrderedDict([(
				"conv0",
				nn.Conv2d(
					in_channels=3,
					out_channels=2*growth_rate,
					kernel_size=(3, 3),
					padding=(1, 1),
					bias=False
				)
			)]))

			channels = 2*growth_rate
			for i in range(1, 4):
				self.features.add_module(
					f"dense_block_{i}",
					DenseNet.DenseBlock(
						in_channels=channels,
						growth_rate=growth_rate,
						layer_count=layers_count_by_block[i - 1],
						dropout_rate=dropout_rate
					)
				)
				channels += layers_count_by_block[i - 1] * growth_rate
				self.features.add_module(
					f"transition_{i}",
					DenseNet.TransitionLayer(
						in_channels=channels,
						compression_factor=compression_factor,
						dropout_rate=dropout_rate
					)
				)
				channels = int(channels * compression_factor)
		else:
			self.features = nn.Sequential(OrderedDict([(
				"conv0",
				nn.Conv2d(
					in_channels=3,
					out_channels=16,
					kernel_size=(3, 3),
					padding=(1, 1),
					bias=False
				)
			)]))


			channels = 16
			for i in range(1, 4):
				self.features.add_module(
					f"dense_block_{i}",
					DenseNet.DenseBlock(
						in_channels=channels,
						growth_rate=growth_rate,
						layer_count=self._layer_count,
						dropout_rate=dropout_rate
					)
				)
				channels += self._layer_count * growth_rate
				self.features.add_module(
					f"transition_{i}",
					DenseNet.TransitionLayer(
						in_channels=channels,
						compression_factor=compression_factor,
						dropout_rate=dropout_rate
					)
				)
				channels = int(channels * compression_factor)

		self._classification_layer = nn.Sequential(
			nn.Linear(in_features=channels, out_features=class_count),
			nn.Softmax(dim=1)
		)

		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.kaiming_normal_(module.weight)
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.constant_(module.weight, 1)
				nn.init.constant_(module.bias, 0)
			elif isinstance(module, nn.Linear):
				nn.init.constant_(module.bias, 0)

	def forward(self, x) -> Tensor:
		assert isinstance(x, Tensor)
		# print("DenseNet::forward")
		# for i, ch in enumerate(x):
		# 	for j, tensor in enumerate(ch):
		# 		image = np.array(tensor.detach().numpy() * 255, dtype=np.uint8)
		# 		cv2.imwrite(f"temp/{self._count}_{i}_{j}_DenseNet_image.png", image)

		x = self.features(x)
		x = F.adaptive_avg_pool2d(x, (1, 1))
		x = torch.flatten(x, 1)
		x = self._classification_layer(x)

		return x
