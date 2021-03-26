import torch
from torch import nn, Tensor
from torch.functional import F
from utils import InceptionConv2d
import gc
import sys


class InceptionV4(nn.Module):
	class Stem(nn.Module):
		def __init__(self):
			super(InceptionV4.Stem, self).__init__()
			self._branch_0 = nn.Sequential(
				InceptionConv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
				InceptionConv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
				InceptionConv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
			)
			self._branch_1a = InceptionConv2d(in_channels=64, out_channels=96, kernel_size=(3, 3), stride=(2, 2))
			self._branch_1b = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
			self._branch_2a = nn.Sequential(
				InceptionConv2d(in_channels=160, out_channels=64, kernel_size=(1, 1), padding=(1, 1)),
				InceptionConv2d(in_channels=64, out_channels=96, kernel_size=(3, 3))
			)
			self._branch_2b = nn.Sequential(
				InceptionConv2d(in_channels=160, out_channels=64, kernel_size=(1, 1), padding=(1, 1)),
				InceptionConv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), padding=(3, 0)),
				InceptionConv2d(in_channels=64, out_channels=64, kernel_size=(1, 7), padding=(0, 3)),
				InceptionConv2d(in_channels=64, out_channels=96, kernel_size=(3, 3))
			)
			self._branch_3a = InceptionConv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=(2, 2))
			self._branch_3b = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

		def forward(self, x: Tensor):
			x = self._branch_0(x)
			x_1a = self._branch_1a(x)
			x_1b = self._branch_1b(x)
			x = torch.cat((x_1a, x_1b), 1)
			del x_1a
			del x_1b
			gc.collect()

			x_2a = self._branch_2a(x)
			conv2d = InceptionConv2d(in_channels=160, out_channels=64, kernel_size=(1, 1), padding=(1, 1))
			x_2b_0: Tensor = conv2d(x)
			del conv2d
			del x
			gc.collect()
			conv2d = InceptionConv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), padding=(3, 0))
			x_2b_1 = conv2d(x_2b_0)
			del conv2d
			del x_2b_0
			gc.collect()
			conv2d = InceptionConv2d(in_channels=64, out_channels=64, kernel_size=(1, 7), padding=(0, 3))
			x_2b_2 = conv2d(x_2b_1)
			del conv2d
			del x_2b_1
			gc.collect()
			sys.getsizeof(x_2b_2)
			conv2d = InceptionConv2d(in_channels=64, out_channels=96, kernel_size=(3, 3))
			x_2b = conv2d(x_2b_2)
			del x_2b_2
			del conv2d
			gc.collect()
			x = torch.cat((x_2a, x_2b), 1)
			del x_2a
			del x_2b
			gc.collect()
			x_3a = self._branch_3a(x)
			x_3b = self._branch_3b(x)
			return torch.cat((x_3a, x_3b), 1)

	class InceptionA(nn.Module):
		def __init__(self):
			super(InceptionV4.InceptionA, self).__init__()
			self._branch_0 = nn.Sequential(
				nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
				InceptionConv2d(in_channels=384, out_channels=96, kernel_size=(1, 1))
			)
			self._branch_1 = InceptionConv2d(in_channels=384, out_channels=96, kernel_size=(1, 1))

			self._branch_2 = nn.Sequential(
				InceptionConv2d(in_channels=384, out_channels=64, kernel_size=(1, 1)),
				InceptionConv2d(in_channels=64, out_channels=96, kernel_size=(3, 3), padding=(1, 1))
			)
			self._branch_3 = nn.Sequential(
				InceptionConv2d(in_channels=384, out_channels=64, kernel_size=(1, 1)),
				InceptionConv2d(in_channels=64, out_channels=96, kernel_size=(3, 3), padding=(1, 1)),
				InceptionConv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), padding=(1, 1))
			)

		def forward(self, x: Tensor) -> torch.Tensor:
			x_0 = self._branch_0(x)
			x_1 = self._branch_1(x)
			x_2 = self._branch_2(x)
			x_3 = self._branch_3(x)
			x = torch.cat((x_0, x_1, x_2, x_3), 1)

			return x

	class ReductionA(nn.Module):
		def __init__(
				self,
				in_channels: int = 384,
				third_branch_first_channels: int = 192,
				third_branch_second_channels: int = 224,
				third_branch_third_channels: int = 256,
				second_branch_channels: int = 384,
				use_batch_norm: bool = True
		):
			super(InceptionV4.ReductionA, self).__init__()
			self._branch_0 = nn.MaxPool2d(kernel_size=3, stride=2)
			self._branch_1 = InceptionConv2d(
				in_channels=in_channels,
				out_channels=second_branch_channels,
				kernel_size=(3, 3),
				stride=(2, 2),
				use_batch_norm=use_batch_norm
			)
			self._branch_2 = nn.Sequential(
				InceptionConv2d(
					in_channels=in_channels,
					out_channels=third_branch_first_channels,
					kernel_size=(1, 1),
					use_batch_norm=use_batch_norm
				),
				InceptionConv2d(
					in_channels=third_branch_first_channels,
					out_channels=third_branch_second_channels,
					kernel_size=(3, 3),
					padding=(1, 1),
					use_batch_norm=use_batch_norm
				),
				InceptionConv2d(
					in_channels=third_branch_second_channels,
					out_channels=third_branch_third_channels,
					kernel_size=(3, 3),
					stride=(2, 2),
					use_batch_norm=use_batch_norm
				)
			)

		def forward(self, x: Tensor) -> Tensor:
			x_0 = self._branch_0(x)
			x_1 = self._branch_1(x)
			x_2 = self._branch_2(x)
			x = torch.cat((x_0, x_1, x_2), 1)

			return x

	class InceptionB(nn.Module):
		def __init__(self):
			super(InceptionV4.InceptionB, self).__init__()
			self._branch_0 = nn.Sequential(
				nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
				InceptionConv2d(in_channels=1024, out_channels=128, kernel_size=(1, 1))
			)
			self._branch_1 = InceptionConv2d(in_channels=1024, out_channels=384, kernel_size=(1, 1))
			self._branch_2 = nn.Sequential(
				InceptionConv2d(in_channels=1024, out_channels=192, kernel_size=(1, 1)),
				InceptionConv2d(in_channels=192, out_channels=224, kernel_size=(1, 7), padding=(0, 3)),
				InceptionConv2d(in_channels=224, out_channels=256, kernel_size=(7, 1), padding=(3, 0))
			)
			self._branch_3 = nn.Sequential(
				InceptionConv2d(in_channels=1024, out_channels=192, kernel_size=(1, 1)),
				InceptionConv2d(in_channels=192, out_channels=192, kernel_size=(1, 7), padding=(0, 3)),
				InceptionConv2d(in_channels=192, out_channels=224, kernel_size=(7, 1), padding=(3, 0)),
				InceptionConv2d(in_channels=224, out_channels=224, kernel_size=(1, 7), padding=(0, 3)),
				InceptionConv2d(in_channels=224, out_channels=256, kernel_size=(7, 1), padding=(3, 0))
			)

		def forward(self, x: torch.Tensor) -> torch.Tensor:
			x_0 = self._branch_0(x)
			x_1 = self._branch_1(x)
			x_2 = self._branch_2(x)
			x_3 = self._branch_3(x)
			x = torch.cat((x_0, x_1, x_2, x_3), 1)

			return x

	class ReductionB(nn.Module):
		def __init__(self):
			super(InceptionV4.ReductionB, self).__init__()
			self._branch_0 = nn.MaxPool2d(kernel_size=3, stride=2)
			self._branch_1 = nn.Sequential(
				InceptionConv2d(in_channels=1024, out_channels=192, kernel_size=(1, 1)),
				InceptionConv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=(2, 2))
			)
			self._branch_2 = nn.Sequential(
				InceptionConv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1)),
				InceptionConv2d(in_channels=256, out_channels=256, kernel_size=(1, 7), padding=(0, 3)),
				InceptionConv2d(in_channels=256, out_channels=320, kernel_size=(7, 1), padding=(3, 0)),
				InceptionConv2d(in_channels=320, out_channels=320, kernel_size=(3, 3), stride=(2, 2))
			)

		def forward(self, x: Tensor) -> Tensor:
			x_0 = self._branch_0(x)
			x_1 = self._branch_1(x)
			x_2 = self._branch_2(x)
			x = torch.cat((x_0, x_1, x_2), 1)

			return x

	class InceptionC(nn.Module):
		def __init__(self):
			super(InceptionV4.InceptionC, self).__init__()
			self._branch_0 = nn.Sequential(
				nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
				InceptionConv2d(in_channels=1536, out_channels=256, kernel_size=(1, 1))
			)
			self._branch_1 = InceptionConv2d(in_channels=1536, out_channels=256, kernel_size=(1, 1))
			self._branch_2_0 = InceptionConv2d(in_channels=1536, out_channels=384, kernel_size=(1, 1))
			self._branch_2_1a = InceptionConv2d(in_channels=384, out_channels=256, kernel_size=(1, 3), padding=(0, 1))
			self._branch_2_1b = InceptionConv2d(in_channels=384, out_channels=256, kernel_size=(3, 1), padding=(1, 0))
			self._branch_3_0 = nn.Sequential(
				InceptionConv2d(in_channels=1536, out_channels=384, kernel_size=(1, 1)),
				InceptionConv2d(in_channels=384, out_channels=448, kernel_size=(1, 3), padding=(0, 1)),
				InceptionConv2d(in_channels=448, out_channels=512, kernel_size=(3, 1), padding=(1, 0))
			)
			self._branch_3_1a = InceptionConv2d(in_channels=512, out_channels=256, kernel_size=(3, 1), padding=(1, 0))
			self._branch_3_1b = InceptionConv2d(in_channels=512, out_channels=256, kernel_size=(1, 3), padding=(0, 1))

		def forward(self, x: torch.Tensor) -> torch.Tensor:
			x_0 = self._branch_0(x)
			x_1 = self._branch_1(x)
			x_2_0 = self._branch_2_0(x)
			x_2_1a = self._branch_2_1a(x_2_0)
			x_2_1b = self._branch_2_1b(x_2_0)
			x_2 = torch.cat((x_2_1a, x_2_1b), 1)
			x_3_0 = self._branch_3_0(x)
			x_3_1a = self._branch_3_1a(x_3_0)
			x_3_1b = self._branch_3_1b(x_3_0)
			x_3 = torch.cat((x_3_1a, x_3_1b), 1)
			x = torch.cat((x_0, x_1, x_2, x_3), 1)

			return x

	def __init__(self, num_classes: int, device: str = "cpu"):
		assert device == "cpu" or device == "cuda"
		super(InceptionV4, self).__init__()
		self.num_classes = num_classes
		self.device = device
		self._architecture = nn.Sequential(
			InceptionV4.Stem(),
			InceptionV4.InceptionA(),
			InceptionV4.InceptionA(),
			InceptionV4.InceptionA(),
			InceptionV4.InceptionA(),
			InceptionV4.ReductionA(),
			InceptionV4.InceptionB(),
			InceptionV4.InceptionB(),
			InceptionV4.InceptionB(),
			InceptionV4.InceptionB(),
			InceptionV4.InceptionB(),
			InceptionV4.InceptionB(),
			InceptionV4.InceptionB(),
			InceptionV4.ReductionB(),
			InceptionV4.InceptionC(),
			InceptionV4.InceptionC(),
			InceptionV4.InceptionC()
		)
		self._last_linear_layer = nn.Linear(1536, self.num_classes)
		self._dropout = nn.Dropout2d(p=0.8)
		self._softmax = nn.Softmax(dim=1)

	def forward(self, x):
		x = self._architecture(x)
		x = F.avg_pool2d(x, kernel_size=x.shape[2])
		x = x.view(x.size(0), -1)
		x = self._dropout(x)
		x = self._last_linear_layer(x)
		x = self._softmax(x)

		return x
