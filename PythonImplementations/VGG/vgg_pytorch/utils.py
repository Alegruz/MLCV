import collections

import torch.utils.model_zoo as model_zoo

GLOBAL_PARAMS = collections.namedtuple("GLOBAL_PARAMS", [
	"configure", "image_size", "batch_norm", "dropout_rate", "num_classes"
])

GLOBAL_PARAMS.__new__.__defaults__ = (None,) * len(GLOBAL_PARAMS._fields)

g_params_dict = {
	"VGG11": ('A', 224, False),
	"VGG13": ('B', 224, False),
	"VGG16": ('D', 224, False),
	"VGG19": ('E', 224, False),
	"VGG11_BN": ('A', 224, True),
	"VGG13_BN": ('B', 224, True),
	"VGG16_BN": ('D', 224, True),
	"VGG19_BN": ('E', 224, True)
}

def vgg_params(model_name):
	return g_params_dict[model_name]


def vggnet(configure, image_size, batch_norm, dropout_rate=0.2, num_classes=1000):
    global_params = GLOBAL_PARAMS(
        configure=configure,
        image_size=image_size,
        batch_norm=batch_norm,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
    )

    return global_params

def get_model_params(model_name, override_params):
    if model_name.startswith("VGG"):
        c, s, b = vgg_params(model_name)
        global_params = vggnet(configure=c, image_size=s, batch_norm=b)
    else:
        raise NotImplementedError(f"model name is not pre-defined: {model_name}.")
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return global_params

urls_map = {
    "VGG11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "VGG13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "VGG16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "VGG19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "VGG11_BN": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "VGG13_BN": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "VGG16_BN": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "VGG19_BN": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


def load_pretrained_weights(model, model_name, load_fc=True):
    """ Loads pretrained weights, and downloads if loading for the first time. """

    state_dict = model_zoo.load_url(urls_map[model_name])
    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop("classifier.6.weight")
        state_dict.pop("classifier.6.bias")
        res = model.load_state_dict(state_dict, strict=False)
        assert set(res.missing_keys) == {"classifier.6.weight", "classifier.6.bias"}, "issue loading pretrained weights"
    print(f"Loaded pretrained weights for {model_name}")