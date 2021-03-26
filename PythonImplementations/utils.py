import torch
from torch import nn, Tensor
from torch.functional import F
from typing import Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"


class InceptionConv2d(nn.Module):
	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			kernel_size: Tuple[int, int],
			stride: Tuple[int, int] = (1, 1),
			padding: Tuple[int, int] = (0, 0),
			dilation: Tuple[int, int] = (1, 1),
			groups: int = 1,
			bias: bool = True,
			padding_mode: str = "zeros",
			eps: float = 1e-5,
			momentum: float = 0.1,
			affine: bool = True,
			track_running_stats: bool = True,
			inplace: bool = True,
			use_batch_norm: bool = True
	):
		super(InceptionConv2d, self).__init__()
		self._conv_2d = nn.Conv2d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=kernel_size,
			stride=stride,
			padding=padding,
			dilation=dilation,
			groups=groups,
			bias=bias,
			padding_mode=padding_mode
		)
		self._use_batch_norm = use_batch_norm
		if self._use_batch_norm:
			self._batch_normalization = nn.BatchNorm2d(
				num_features=out_channels,
				eps=eps,
				momentum=momentum,
				affine=affine,
				track_running_stats=track_running_stats
			)
		self._relu = nn.ReLU(
			inplace=inplace
		)

	def forward(self, x):
		x = self._conv_2d(x)
		if self._use_batch_norm:
			x = self._batch_normalization(x)
		x = self._relu(x)

		return x


class InceptionResNetConv2d(InceptionConv2d):
	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			kernel_size: Tuple[int, int],
			stride: Tuple[int, int] = (1, 1),
			padding: Tuple[int, int] = (0, 0),
			dilation: Tuple[int, int] = (1, 1),
			groups: int = 1,
			bias: bool = False,
			padding_mode: str = "zeros",
			eps: float = 0.001,
			momentum: float = 0.1,
			affine: bool = True,
			track_running_stats: bool = True,
			inplace: bool = False,
			use_batch_norm: bool = True
	):
		super(InceptionResNetConv2d, self).__init__(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=kernel_size,
			stride=stride,
			padding=padding,
			dilation=dilation,
			groups=groups,
			bias=bias,
			padding_mode=padding_mode,
			eps=eps,
			momentum=momentum,
			affine=affine,
			track_running_stats=track_running_stats,
			inplace=inplace,
			use_batch_norm=use_batch_norm
		)


class DenseNetConv2d(nn.Module):
	def __init__(
			self,
			in_channels: int,
			growth_rate: int,
			kernel_size: Tuple[int, int],
			is_bottleneck: bool = False,
			stride: Tuple[int, int] = (1, 1),
			padding: Tuple[int, int] = (0, 0),
			eps: float = 1e-5,
			momentum: float = 1.0,
			affine: bool = True,
			track_running_stats: bool = True,
			inplace: bool = True,
			dilation: Tuple[int, int] = (1, 1),
			groups: int = 1,
			bias: bool = False,
			dropout_rate: float = 0.0
	):
		super(DenseNetConv2d, self).__init__()
		self._dropout_rate = dropout_rate
		if is_bottleneck:
			self._architecture = nn.Sequential(
				nn.BatchNorm2d(
					num_features=in_channels,
					eps=eps,
					momentum=momentum,
					affine=affine,
					track_running_stats=track_running_stats
				),
				nn.ReLU(inplace=inplace),
				nn.Conv2d(
					in_channels=in_channels,
					out_channels=4 * growth_rate,
					kernel_size=(1, 1),
					dilation=dilation,
					groups=groups,
					bias=bias
				),
				nn.Dropout2d(p=dropout_rate),
				nn.BatchNorm2d(
					num_features=4 * growth_rate,
					eps=eps,
					momentum=momentum,
					affine=affine,
					track_running_stats=track_running_stats
				),
				nn.ReLU(inplace=inplace),
				nn.Conv2d(
					in_channels=4 * growth_rate,
					out_channels=growth_rate,
					kernel_size=(3, 3),
					stride=(1, 1),
					padding=(1, 1),
					dilation=dilation,
					groups=groups,
					bias=bias
				),
				nn.Dropout2d(p=dropout_rate)
			)
		else:
			self.add_module(
				"norm0",
				nn.BatchNorm2d(
					num_features=in_channels,
					eps=eps,
					momentum=momentum,
					affine=affine,
					track_running_stats=track_running_stats
				).to(device)
			)
			self.add_module("relu0", nn.ReLU(inplace=inplace))
			self.add_module(
				"conv0",
				nn.Conv2d(
					in_channels=in_channels,
					out_channels=growth_rate,
					kernel_size=kernel_size,
					stride=stride,
					padding=padding,
					dilation=dilation,
					groups=groups,
					bias=bias
				)
			)

	def forward(self, x: Tensor):
		assert isinstance(x, Tensor)

		x = self.conv0(self.relu0(self.norm0(x)))
		if self._dropout_rate > 0.0:
			x = F.dropout(input=x, p=self._dropout_rate, training=self.training)

		return x


class Fire(nn.Module):
	def __init__(
			self,
			in_channels: int,
			squeeze_filter_count: int,
			expand_1x1_filter_count: int,
			expand_3x3_filter_count: int
	):
		super(Fire, self).__init__()
		self._squeeze = nn.Conv2d(in_channels=in_channels, out_channels=squeeze_filter_count, kernel_size=(1, 1))
		self._squeeze_activation = nn.ReLU(inplace=True)
		self._expand_1x1 = nn.Conv2d(
			in_channels=squeeze_filter_count,
			out_channels=expand_1x1_filter_count,
			kernel_size=(1, 1)
		)
		self._expand_1x1_activation = nn.ReLU(inplace=True)
		self._expand_3x3 = nn.Conv2d(
			in_channels=squeeze_filter_count,
			out_channels=expand_3x3_filter_count,
			kernel_size=(3, 3),
			padding=(1, 1)
		)
		self._expand_3x3_activation = nn.ReLU(inplace=True)

	def forward(self, x: Tensor) -> Tensor:
		x = self._squeeze_activation(self._squeeze(x))
		return torch.cat(
			[
				self._expand_1x1_activation(self._expand_1x1(x)),
				self._expand_3x3_activation(self._expand_3x3(x))
			],
			1
		)
