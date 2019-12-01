import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import os


#sizes
human_attributes_size = 18
encoded_description_size = 100
flatten_down_sampled_segmentations_size = 256
gausian_noise_size = 100

design_encoding = human_attributes_size + encoded_description_size + gausian_noise_size # 218



class Generator1(nn.Module):
    def __init__(self):
        super(Generator1,self).__init__()

        self.G1_Layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=design_encoding,out_channels=1024,kernel_size=4,stride=4,padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=4,stride=2,padding=1),
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
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x_design_desc, x_down_sampled_image):
        x_design = self.G1_Layer2(x_design_desc)
        x_down_sampled = self.G1_A(x_down_sampled_image)
        concatenated = torch.cat((x_design,x_down_sampled),1)
        return self.G1_LastLayers(concatenated)

class Generator2(nn.Module):
    def __init__(self):
        super(Generator2,self).__init__()
        self.G2_Layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=design_encoding,out_channels=1024,kernel_size=4,stride=4,padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.G2_LayerC = nn.Sequential(
            nn.Conv2d(in_channels=7,out_channels=64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.G2_Layer6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1),            
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, x_design_desc, s_tilde):
        l3 = self.G2_Layer3(x_design_desc)
        lc = self.G2_LayerC(s_tilde)
        concatenated = torch.cat((l3,lc),dim=1)
        return self.G2_Layer6(concatenated)



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
            nn.Conv2d(in_channels=4,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.D1_concatenation = nn.Sequential(
            nn.Conv2d(in_channels=640,out_channels=1024,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )

        self.D1_LastLayers = nn.Sequential(
            nn.Conv2d(in_channels=1142,out_channels=1024,kernel_size=1,stride=1,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=1024,out_channels=1,kernel_size=4,stride=4,padding=0),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_segmented_image,x_down_sampled, x_design_encoding):
        layer4 = self.D1_Layer1(x_segmented_image)
        x_down_sampled = self.D1_condition(x_down_sampled)
        
        concatenated = torch.cat((layer4,x_down_sampled),1)
        layer5 = self.D1_concatenation(concatenated)
        d_by_4 = x_design_encoding.repeat(1,16)
        d_by_4 = d_by_4.view(layer5.shape[0],x_design_encoding.shape[1],4,4)
        input_for_layer6 = torch.cat((layer5,d_by_4),1)
        output = self.D1_LastLayers(input_for_layer6)
        output = output.view(x_segmented_image.shape[0],1)
        return self.sigmoid(output)



class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2,self).__init__()
        self.D2_Layer3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)            
        )

        self.D2_LayerC = nn.Sequential(
            nn.Conv2d(in_channels=7,out_channels=64,kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),            
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),            
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
            
        )

        self.D2_Layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)         
        )

        self.D2_Layer7 = nn.Sequential(
            nn.Conv2d(in_channels=1142,out_channels=1024,kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(1024),
            nn.Conv2d(in_channels=1024,out_channels=1,kernel_size=4, stride=4, padding=0),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, design_encoding, g2_output, s_tilde):
        l3 = self.D2_Layer3(g2_output)
        lc = self.D2_LayerC(s_tilde)
        concatenated = torch.cat((l3,lc),dim=1)
        l5 = self.D2_Layer5(concatenated)
        d_by_4 = design_encoding.repeat(1,16)
        d_by_4 = d_by_4.view(l5.shape[0],design_encoding.shape[1],4,4)
        input_for_l7 = torch.cat((l5,d_by_4),1)
        l7 = self.D2_Layer7(input_for_l7)
        return self.sigmoid(l7)




        