from thop import profile
from torchvision.models import resnet50
import torchvision.models as models
import torch
from flopth import flopth

model1 = models.alexnet()
model2=models.vgg16()
model3=models.resnet101()
model4=models.inception_v3()
model5=models.densenet161()
input=torch.randn(3,299,299)
print("Total number of paramerters in alexnet is {}  ".format(sum(x.numel() for x in model1.parameters())))
print("Total number of paramerters in vgg16 is {}  ".format(sum(x.numel() for x in model2.parameters())))
print("Total number of paramerters in resnet101 is {}  ".format(sum(x.numel() for x in model3.parameters())))
print("Total number of paramerters in inception_v3 is {}  ".format(sum(x.numel() for x in model4.parameters())))
print("Total number of paramerters in densenet161 is {}  ".format(sum(x.numel() for x in model5.parameters())))
print("alexnet: ",flopth(model1, in_size=[3,299,299]),)
print("vgg16: ",flopth(model2, in_size=[3,299,299]))
print("resnet101: ",flopth(model3, in_size=[3,299,299]))
print("inception_v3: ",flopth(model4, in_size=[3,299,299]))
print("densenet161: ",flopth(model5, in_size=[3,299,299]))
