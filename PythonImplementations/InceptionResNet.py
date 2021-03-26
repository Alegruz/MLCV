import torch
from torch import nn, Tensor
from torch.functional import F
from utils import InceptionResNetConv2d
from InceptionV4 import InceptionV4
import time


class InceptionResNet(nn.Module):
	class Stem(nn.Module):
		def __init__(self):
			super(InceptionResNet.Stem, self).__init__()
			self._stack = None

		def forward(self, x):
			x = self._stack(x)

			return x

	class InceptionResNetA(nn.Module):
		def __init__(self):
			super(InceptionResNet.InceptionResNetA, self).__init__()
			self._stack = None

		def forward(self, x):
			x = self._stack(x)

			return x

	class InceptionResNetB(nn.Module):
		def __init__(self):
			super(InceptionResNet.InceptionResNetB, self).__init__()
			self._stack = None

		def forward(self, x):
			x = self._stack(x)

			return x

	class InceptionResNetC(nn.Module):
		def __init__(self):
			super(InceptionResNet.InceptionResNetC, self).__init__()
			self._stack = None

		def forward(self, x):
			x = self._stack(x)

			return x

	class ReductionB(nn.Module):
		def __init__(self):
			super(InceptionResNet.ReductionB, self).__init__()

	def __init__(
			self,
			num_classes: int,
			stem: Stem,
			inception_resnet_a: InceptionResNetA,
			inception_resnet_b: InceptionResNetB,
			reduction_b: ReductionB,
			inception_resnet_c: InceptionResNetC,
			in_channels: int,
			out_channels: int,
			k: int,
			l: int,
			m: int,
			n: int,
			device: str = "cpu"
	):
		assert device == "cpu" or device == "cuda"
		super(InceptionResNet, self).__init__()
		self.num_classes = num_classes
		self.device = device
		self._stem = stem
		self._inception_resnet_a = inception_resnet_a
		self._inception_resnet_b = inception_resnet_b
		self._reduction_b = reduction_b
		self._inception_resnet_c = inception_resnet_c

		self._architecture = nn.Sequential(
			self._stem,
			self._inception_resnet_a,
			self._inception_resnet_a,
			self._inception_resnet_a,
			self._inception_resnet_a,
			self._inception_resnet_a,
			InceptionV4.ReductionA(in_channels, k, l, m, n),
			self._inception_resnet_b,
			self._inception_resnet_b,
			self._inception_resnet_b,
			self._inception_resnet_b,
			self._inception_resnet_b,
			self._inception_resnet_b,
			self._inception_resnet_b,
			self._inception_resnet_b,
			self._inception_resnet_b,
			self._inception_resnet_b,
			self._reduction_b,
			self._inception_resnet_c,
			self._inception_resnet_c,
			self._inception_resnet_c,
			self._inception_resnet_c,
			self._inception_resnet_c
		)
		self.last_linear_layer = nn.Linear(out_channels, self.num_classes)
		self._dropout = nn.Dropout2d(p=0.8)
		self._softmax = nn.Softmax(dim=1)

	def forward(self, x):
		x = self._architecture(x)
		x = F.avg_pool2d(x, kernel_size=x.shape[2])
		x = x.view(x.size(0), -1)
		x = self._dropout(x)
		x = self.last_linear_layer(x)
		x = self._softmax(x)

		return x


class InceptionResNetV1(InceptionResNet):
	class Stem(InceptionResNet.Stem):
		def __init__(self):
			super(InceptionResNetV1.Stem, self).__init__()
			self._stack = nn.Sequential(
				InceptionResNetConv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
				InceptionResNetConv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
				InceptionResNetConv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
				nn.MaxPool2d(kernel_size=3, stride=2),
				InceptionResNetConv2d(in_channels=64, out_channels=80, kernel_size=(1, 1)),
				InceptionResNetConv2d(in_channels=80, out_channels=192, kernel_size=(3, 3)),
				InceptionResNetConv2d(in_channels=192, out_channels=256, kernel_size=(3, 3), stride=(2, 2))
			)

	class InceptionResNetA(InceptionResNet.InceptionResNetA):
		def __init__(self, scale: float = 1.0):
			super(InceptionResNetV1.InceptionResNetA, self).__init__()
			self._scale = scale
			self._branch_0 = InceptionResNetConv2d(in_channels=256, out_channels=32, kernel_size=(1, 1))
			self._branch_1 = nn.Sequential(
				InceptionResNetConv2d(in_channels=256, out_channels=32, kernel_size=(1, 1)),
				InceptionResNetConv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
			)
			self._branch_2 = nn.Sequential(
				InceptionResNetConv2d(in_channels=256, out_channels=32, kernel_size=(1, 1)),
				InceptionResNetConv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1)),
				InceptionResNetConv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
			)
			self._conv2d = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(1, 1))
			self._relu = nn.ReLU(inplace=False)

		def forward(self, x):
			x_0 = self._branch_0(x)
			x_1 = self._branch_1(x)
			x_2 = self._branch_2(x)
			out = torch.cat((x_0, x_1, x_2), 1)
			out = self._conv2d(out)
			out = out * self._scale + x
			out = self._relu(out)

			return out

	class InceptionResNetB(InceptionResNet.InceptionResNetB):
		def __init__(self, scale: float = 1.0):
			super(InceptionResNetV1.InceptionResNetB, self).__init__()
			self._scale = scale
			self._branch_0 = InceptionResNetConv2d(in_channels=896, out_channels=128, kernel_size=(1, 1))
			self._branch_1 = nn.Sequential(
				InceptionResNetConv2d(in_channels=896, out_channels=128, kernel_size=(1, 1)),
				InceptionResNetConv2d(in_channels=128, out_channels=128, kernel_size=(1, 7), padding=(0, 3)),
				InceptionResNetConv2d(in_channels=128, out_channels=128, kernel_size=(7, 1), padding=(3, 0))
			)
			self._conv2d = nn.Conv2d(in_channels=256, out_channels=896, kernel_size=(1, 1))
			self._relu = nn.ReLU(inplace=False)

		def forward(self, x):
			x_0 = self._branch_0(x)
			x_1 = self._branch_1(x)
			out = torch.cat((x_0, x_1), 1)
			out = self._conv2d(out)
			out = out * self._scale + x
			out = self._relu(out)

			return out

	class ReductionB(InceptionResNet.ReductionB):
		def __init__(self):
			super(InceptionResNetV1.ReductionB, self).__init__()
			self._branch_0 = nn.MaxPool2d(kernel_size=3, stride=2)
			self._branch_1 = nn.Sequential(
				InceptionResNetConv2d(in_channels=896, out_channels=256, kernel_size=(1, 1)),
				InceptionResNetConv2d(
					in_channels=256,
					out_channels=384,
					kernel_size=(3, 3),
					stride=(2, 2)
				)
			)
			self._branch_2 = nn.Sequential(
				InceptionResNetConv2d(in_channels=896, out_channels=256, kernel_size=(1, 1)),
				InceptionResNetConv2d(
					in_channels=256,
					out_channels=256,
					kernel_size=(3, 3),
					stride=(2, 2)
				)
			)
			self._branch_3 = nn.Sequential(
				InceptionResNetConv2d(in_channels=896, out_channels=256, kernel_size=(1, 1)),
				InceptionResNetConv2d(
					in_channels=256,
					out_channels=256,
					kernel_size=(3, 3),
					padding=(1, 1)
				),
				InceptionResNetConv2d(
					in_channels=256,
					out_channels=256,
					kernel_size=(3, 3),
					stride=(2, 2)
				)
			)

		def forward(self, x):
			x_0 = self._branch_0(x)
			x_1 = self._branch_1(x)
			x_2 = self._branch_2(x)
			x_3 = self._branch_3(x)
			x = torch.cat((x_0, x_1, x_2, x_3), 1)

			return x

	class InceptionResNetC(InceptionResNet.InceptionResNetC):
		def __init__(self, scale: float = 1.0):
			super(InceptionResNetV1.InceptionResNetC, self).__init__()
			self._scale = scale
			self._branch_0 = InceptionResNetConv2d(in_channels=1792, out_channels=192, kernel_size=(1, 1))
			self._branch_1 = nn.Sequential(
				InceptionResNetConv2d(in_channels=1792, out_channels=192, kernel_size=(1, 1)),
				InceptionResNetConv2d(in_channels=192, out_channels=192, kernel_size=(1, 3), padding=(0, 1)),
				InceptionResNetConv2d(in_channels=192, out_channels=192, kernel_size=(3, 1), padding=(1, 0))
			)
			self._conv2d = nn.Conv2d(in_channels=384, out_channels=1792, kernel_size=(1, 1))
			self._relu = nn.ReLU(inplace=False)

		def forward(self, x):
			x_0 = self._branch_0(x)
			x_1 = self._branch_1(x)
			out = torch.cat((x_0, x_1), 1)
			out = self._conv2d(out)
			out = out * self._scale + x
			out = self._relu(out)

			return out


	def __init__(self, num_classes: int, device: str = "cpu"):
		super(InceptionResNetV1, self).__init__(
			num_classes=num_classes,
			stem=InceptionResNetV1.Stem(),
			inception_resnet_a=InceptionResNetV1.InceptionResNetA(scale=0.17),
			inception_resnet_b=InceptionResNetV1.InceptionResNetB(scale=0.10),
			reduction_b=InceptionResNetV1.ReductionB(),
			inception_resnet_c=InceptionResNetV1.InceptionResNetC(scale=0.20),
			in_channels=256,
			out_channels=1792,
			k=192,
			l=192,
			m=256,
			n=384,
			device=device
		)


class InceptionResNetV2(InceptionResNet):
	class Stem(InceptionResNet.Stem):
		def __init__(self):
			super(InceptionResNetV2.Stem, self).__init__()
			self._branch_0 = nn.Sequential(
				InceptionResNetConv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
				InceptionResNetConv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
				InceptionResNetConv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
			)
			self._branch_1a = InceptionResNetConv2d(in_channels=64, out_channels=96, kernel_size=(3, 3), stride=(2, 2))
			self._branch_1b = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
			self._branch_2a = nn.Sequential(
				InceptionResNetConv2d(in_channels=160, out_channels=64, kernel_size=(1, 1), padding=(1, 1)),
				InceptionResNetConv2d(in_channels=64, out_channels=96, kernel_size=(3, 3))
			)
			self._branch_2b = nn.Sequential(
				InceptionResNetConv2d(in_channels=160, out_channels=64, kernel_size=(1, 1), padding=(1, 1)),
				InceptionResNetConv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), padding=(3, 0)),
				InceptionResNetConv2d(in_channels=64, out_channels=64, kernel_size=(1, 7), padding=(0, 3)),
				InceptionResNetConv2d(in_channels=64, out_channels=96, kernel_size=(3, 3))
			)
			self._branch_3a = InceptionResNetConv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=(2, 2))
			self._branch_3b = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

		def forward(self, x: Tensor):
			x = self._branch_0(x)
			x_1a = self._branch_1a(x)
			x_1b = self._branch_1b(x)
			x = torch.cat((x_1a, x_1b), 1)
			x_2a = self._branch_2a(x)
			x_2b = self._branch_2b(x)
			x = torch.cat((x_2a, x_2b), 1)
			x_3a = self._branch_3a(x)
			x_3b = self._branch_3b(x)
			x = torch.cat((x_3a, x_3b), 1)

			return x

	class InceptionResNetA(InceptionResNet.InceptionResNetA):
		def __init__(self, scale: float = 1.0):
			super(InceptionResNetV2.InceptionResNetA, self).__init__()
			self._scale = scale
			self._branch_0 = InceptionResNetConv2d(in_channels=384, out_channels=32, kernel_size=(1, 1))
			self._branch_1 = nn.Sequential(
				InceptionResNetConv2d(in_channels=384, out_channels=32, kernel_size=(1, 1)),
				InceptionResNetConv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
			)
			self._branch_2 = nn.Sequential(
				InceptionResNetConv2d(in_channels=384, out_channels=32, kernel_size=(1, 1)),
				InceptionResNetConv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), padding=(1, 1)),
				InceptionResNetConv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
			)
			self._conv2d = nn.Conv2d(in_channels=128, out_channels=384, kernel_size=(1, 1))
			self._relu = nn.ReLU(inplace=False)

		def forward(self, x):
			x_0 = self._branch_0(x)
			x_1 = self._branch_1(x)
			x_2 = self._branch_2(x)
			out = torch.cat((x_0, x_1, x_2), 1)
			out = self._conv2d(out)
			out = out * self._scale + x
			out = self._relu(out)
			return out

	class InceptionResNetB(InceptionResNet.InceptionResNetB):
		def __init__(self, scale: float = 1.0):
			super(InceptionResNetV2.InceptionResNetB, self).__init__()
			self._scale = scale
			self._branch_0 = InceptionResNetConv2d(in_channels=1152, out_channels=192, kernel_size=(1, 1))
			self._branch_1 = nn.Sequential(
				InceptionResNetConv2d(in_channels=1152, out_channels=128, kernel_size=(1, 1)),
				InceptionResNetConv2d(in_channels=128, out_channels=160, kernel_size=(1, 7), padding=(0, 3)),
				InceptionResNetConv2d(in_channels=160, out_channels=192, kernel_size=(7, 1), padding=(3, 0))
			)
			self._conv2d = nn.Conv2d(in_channels=384, out_channels=1152, kernel_size=(1, 1))
			self._relu = nn.ReLU(inplace=False)

		def forward(self, x):
			x_0 = self._branch_0(x)
			x_1 = self._branch_1(x)
			out = torch.cat((x_0, x_1), 1)
			out = self._conv2d(out)
			out = out * self._scale + x
			out = self._relu(out)
			return out

	class ReductionB(InceptionResNet.ReductionB):
		def __init__(self):
			super(InceptionResNetV2.ReductionB, self).__init__()
			self._branch_0 = nn.MaxPool2d(kernel_size=3, stride=2)
			self._branch_1 = nn.Sequential(
				InceptionResNetConv2d(in_channels=1152, out_channels=256, kernel_size=(1, 1)),
				InceptionResNetConv2d(
					in_channels=256,
					out_channels=384,
					kernel_size=(3, 3),
					stride=(2, 2)
				)
			)
			self._branch_2 = nn.Sequential(
				InceptionResNetConv2d(in_channels=1152, out_channels=256, kernel_size=(1, 1)),
				InceptionResNetConv2d(
					in_channels=256,
					out_channels=288,
					kernel_size=(3, 3),
					stride=(2, 2)
				)
			)
			self._branch_3 = nn.Sequential(
				InceptionResNetConv2d(in_channels=1152, out_channels=256, kernel_size=(1, 1)),
				InceptionResNetConv2d(
					in_channels=256,
					out_channels=288,
					kernel_size=(3, 3),
					padding=(1, 1)
				),
				InceptionResNetConv2d(
					in_channels=288,
					out_channels=320,
					kernel_size=(3, 3),
					stride=(2, 2)
				)
			)

		def forward(self, x):
			x_0 = self._branch_0(x)
			x_1 = self._branch_1(x)
			x_2 = self._branch_2(x)
			x_3 = self._branch_3(x)
			x = torch.cat((x_0, x_1, x_2, x_3), 1)

			return x

	class InceptionResNetC(InceptionResNet.InceptionResNetC):
		def __init__(self, scale: float = 1.0):
			super(InceptionResNetV2.InceptionResNetC, self).__init__()
			self._scale = scale
			self._branch_0 = InceptionResNetConv2d(in_channels=2144, out_channels=192, kernel_size=(1, 1))
			self._branch_1 = nn.Sequential(
				InceptionResNetConv2d(in_channels=2144, out_channels=192, kernel_size=(1, 1)),
				InceptionResNetConv2d(in_channels=192, out_channels=224, kernel_size=(1, 3), padding=(0, 1)),
				InceptionResNetConv2d(in_channels=224, out_channels=256, kernel_size=(3, 1), padding=(1, 0))
			)
			self._conv2d = nn.Conv2d(in_channels=448, out_channels=2144, kernel_size=(1, 1))
			self._relu = nn.ReLU(inplace=False)

		def forward(self, x):
			x_0 = self._branch_0(x)
			x_1 = self._branch_1(x)
			out = torch.cat((x_0, x_1), 1)
			out = self._conv2d(out)
			out = out * self._scale + x
			out = self._relu(out)
			return out

	def __init__(self, num_classes: int, device: str = "cpu"):
		super(InceptionResNetV2, self).__init__(
			num_classes=num_classes,
			stem=InceptionResNetV2.Stem(),
			inception_resnet_a=InceptionResNetV2.InceptionResNetA(scale=0.17),
			inception_resnet_b=InceptionResNetV2.InceptionResNetB(scale=0.10),
			reduction_b=InceptionResNetV2.ReductionB(),
			inception_resnet_c=InceptionResNetV2.InceptionResNetC(scale=0.20),
			in_channels=384,
			out_channels=2144,
			k=256,
			l=256,
			m=384,
			n=384,
			device=device
		)
