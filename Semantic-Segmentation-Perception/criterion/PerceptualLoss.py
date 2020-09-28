import torch
import torch.nn as nn
from torchvision.models.vgg import vgg19
from torchvision import models
from torch.autograd import Variable



class VggLoss(nn.Module):
    def __init__(self):
        super(VggLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        vgg.features[0] = nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        vgg.cuda()
        loss_network = nn.Sequential(*list(vgg.features)).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.L1Loss()

    def forward(self, out_images, target_images):
        #out_images = torch.cat((out_images, out_images, out_images), 1)
        target_images = torch.cat((target_images, target_images, target_images,target_images, target_images, target_images), 1)
        #target_images = torch.cat((target_images, target_images), 1)

        out_images=self.loss_network(out_images)
        #print(out_images.size())
        #print(type(out_images))
        target_images=target_images.float()
        target_images=self.loss_network(target_images)
        #print(target_images.size())
        perception_loss = self.mse_loss(out_images,target_images)

        return perception_loss
