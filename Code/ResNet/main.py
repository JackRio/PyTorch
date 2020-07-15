import torch
import os

from torchvision import models, transforms
from PIL import Image
from Code.config import config

"""
    Residual Network (ResNet):
        ResNet beat several benchmark in year 2015 by achieving stable training in depth.
"""

# Pulls a pre-trained model that trained the ResNet on 101 layers and on ImageNet Dataset
resnet = models.resnet101(pretrained=True)
print("Layers in the Model\n", resnet)

# Things in the pipeline below
# Resize: Scale the image to 256,256
# CenterCrop: Crop from center into 224,224 size
# ToTensor: Convert image into PyTorch multidimensional array (3D in this case)
# Normalize: Normalize the data as it was done during training

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

img = Image.open(os.path.join(config.PATH, "data/p1ch2/bobby.jpg"))
# img.show()

# Passing the image through the preprocess pipeline defined above
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

# To make the prediction the model should be in eval mode
resnet.eval()

out = resnet(batch_t)
print("Output tensor:", out)

# Mapping the label for the image based on the predictions
with open(os.path.join(config.PATH, "data/p1ch2/imagenet_classes.txt")) as text:
    labels = [line.strip() for line in text.readlines()]

index = torch.max(out, 1)[1]

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print("Predicting the class of the image", labels[index[0]], percentage[index[0]].item())

# Use the sort function to find the remaining class
_, indices = torch.sort(out, descending=True)
print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])
