import torch
import torch.nn as nn

from .utils import get_model_params
from .utils import load_pretrained_weights
from .utils import vgg_params

configures = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

class Vgg(nn.Module):
	def __init__(self, global_params=None):
		super(Vgg, self).__init__()

		self.features = make_layers(configures[global_params.configure], global_params.batch_norm)
		self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(global_params.dropout_rate),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(global_params.dropout_rate),
			nn.Linear(4096, global_params.num_classes)
		)

		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.kaiming_normal_(module.weight, mode = "fan_out", nonlinearity = "relu")
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.constant_(module.weight, 1)
				nn.init.constant_(module.bias, 0)
			elif isinstance(module, nn.Linear):
				nn.init.normal_(module.weight, 0, 0.01)
				nn.init.constant_(module.bias, 0)


	def extract_features(self, inputs):
		return self.features(inputs)

	def forward(self, input):
		input = self.features(input)
		input = self.avg_pool(input)
		input = torch.flatten(input, 1)
		return self.classifier(input)

	@classmethod
	def from_name(cls, model_name, override_params=None):
		cls._check_model_name_is_valid(model_name)
		global_params = get_model_params(model_name, override_params)
		return cls(global_params)

	@classmethod
	def from_pretrained(cls, model_name, num_classes=1000):
		model = cls.from_name(model_name, override_params={"num_classes": num_classes})
		load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
		return model

	@classmethod
	def get_image_size(cls, model_name):
		cls._check_model_name_is_valid(model_name)
		_, res, _ = vgg_params(model_name)
		return res

	@classmethod
	def _check_model_name_is_valid(cls, model_name):
		valid_models = ['VGG' + str(i) for i in ["11", "11_BN",
													"13", "13_BN",
													"16", "16_BN",
													"19", "19_BN"]]
		if model_name not in valid_models:
			raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

def make_layers(configures, batch_norm):
	layers = []
	in_channels = 3
	for configure in configures:
		if configure == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, configure, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(configure), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = configure
	return nn.Sequential(*layers)
