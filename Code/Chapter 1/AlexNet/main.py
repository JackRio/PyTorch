# Checking the pre-trained models within PyTorch for Vision projects
from torchvision import models

"""
    Notes: 
        a. Capitalize names refer to Python classes that implement a number of models.
        b. Lower case names refer to one instance of those main classes.
        
        Ex: resnet101 returns the instance of ResNet with 101 layers.
"""
print("Models ", dir(models))

# AlexNet model intiaization

"""
    alexnet variable below doesn't have any weights assigned (i.e It is not yet trained) if we use 
    this variable to predict the output will be random answers.
    
    We can use pre-trained models which will have weights and other parameters already defined to
    give accurate results.
"""

alexnet = models.AlexNet()
