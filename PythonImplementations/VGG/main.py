import json

import torch
import torchvision.transforms as transforms
from PIL import Image

from vgg_pytorch import Vgg

# Open image
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

input_image = Image.open(filename)

# Preprocess image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# Load class names
# labels_map = json.load(open("labels_map.txt"))
# labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify with VGG11
model = Vgg.from_pretrained("VGG11")
model.eval()

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
	input_batch = input_batch.to("cuda")
	model.to("cuda")

with torch.no_grad():
	logits = model(input_batch)
preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

print("-----")
for idx in preds:
	# label = labels_map[idx]
	prob = torch.softmax(logits, dim=1)[0, idx].item()
	print(f"{idx} ({prob * 100:.2f}%)")