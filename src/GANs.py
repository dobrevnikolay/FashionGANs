import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import os

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

#todo: calculate the output size of each layer, foward, training loop
# def one_hot(labels):
#     y = torch.eye(len(classes)) 
#     return y[labels]

#sizes
human_attributes_size = 8
encoded_description_size = 100
flatten_down_sampled_segmentations_size = 256
gausian_noise_size = 100

design_encoding = human_attributes_size + encoded_description_size + gausian_noise_size # 208


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Generator1(nn.Module):
    def __init__(self):
        super(Generator1, self).__init__()

        self.G1_Layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=design_encoding,out_channels=1024,kernel_size=4,stride=4),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=4,stride=2,padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.G1_A = nn.Sequential(
            nn.Conv2d(in_channels=4,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.G1_LastLayers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=(128+512),out_channels=256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64,out_channels=7,kernel_size=4,stride=2,padding=1),
            nn.SoftMax()
        )

    def forward(self, x_design_desc, x_down_sampled_image):
        x_design = self.G1_Layer2(x_design_desc)
        x_down_sampled = self.G1_Layer2(x_down_sampled_image)
        concatenated = torch.cat((Flatten(x_design),Flatten(x_down_sampled)),0)
        return self.G1_LastLayers(concatenated)

        
class Discriminator1(nn.Module):
    def __init__(self):
        super(Discriminator1,self).__init__()

        self.D1_Layer1 = nn.Sequential(
            nn.Conv2d(in_channels=7,out_channels=64,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )

        self.D1_condition = nn.Sequential(
            nn.Conv2d(in_channels=4,out_channels=64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.D1_concatenation = nn.Sequential(
            nn.Conv2d(in_channels=640,out_channels=1024,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )

        self.D1_LastLayers = nn.Sequential(
            nn.Conv2d(in_channels=1132,out_channels=1024,kernel_size=1,stride=1,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=1024,out_channels=1,kernel_size=4,stride=4,padding=0),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x_segmented_image,x_down_sampled, x_design_encoding):
        layer4 = self.D1_Layer1(x_segmented_image)
        x_down_sampled = self.D1_condition(x_down_sampled)
        layer5 = self.D1_concatenation(torch.cat((Flatten(layer4),Flatten(x_down_sampled)),0))
        d_by_4 = x_design_encoding.repeat(1,4)
        input_for_layer6 = torch.cat((layer5,d_by_4),0)
        return self.D1_LastLayers(input_for_layer6)
        