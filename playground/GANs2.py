import torch
import torch.nn as nn
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import ToTensor
from functools import reduce

#todo: calculate the output size of each layer, foward, training loop
def one_hot(labels):
    y = torch.eye(len(classes)) 
    return y[labels]

class G1(nn.Module):
    def __init__(self):
        super(G1, self).__init__()

        self.G1_1 = nn.ConvTranspose2d(in_channels =150, out_channels =1024, kernel_size=4, stride=4)
        self.G1_1_B = nn.BatchNorm2d(1024)

        self.G1_2 = nn.ConvTranspose2d(in_channels =512, out_channels =512, kernel_size=4, stride=2, padding=1)
        self.G1_2_B = nn.BatchNorm2d(512)

        self.G1AB = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
            ).to(device)

        self.G1_3 = nn.ConvTranspose2d(in_channels =640, out_channels =256, kernel_size=4, stride=2, padding=1)
        self.G1_3_B = nn.BatchNorm2d(256)

        self.G1_4 = nn.ConvTranspose2d(in_channels =256, out_channels =128, kernel_size=4, stride=2, padding=1)
        self.G1_4_B = nn.BatchNorm2d(128)

        self.G1_5 = nn.ConvTranspose2d(in_channels =128, out_channels =64, kernel_size=4, stride=2, padding=1)
        self.G1_5_B = nn.BatchNorm2d(64)

        self.G1_6 = nn.ConvTranspose2d(in_channels =64, out_channels =7, kernel_size=4, stride=2, padding=1)


        


class G2(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.G2_123 = nn.Sequential(
            nn.ConvTranspose2d(in_channels =(100+108), out_channels =1024, kernel_size=4, stride=4),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels =1024, out_channels =512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels =512, out_channels =256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ).to(device)

        self.G2_ABC = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ).to(device)

        self.G2_456 = nn.sequential(
            nn.ConvTranspose2d(in_channels =512, out_channels =128, kernel_size=4, stride=2, padding=1),            
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels =128, out_channels =64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels =64, out_channels =3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        ).to(device)
    def forward(self, x_design_desc, x_segmented_image):
        O3 = self.G2_123(x_design_desc)
        OC = self.G2ABC(x_down_sampled_image)
        # O6 = torch.cat((Flatten(x_design),Flatten(x_down_sampled)),0)
        O6 = self.G2_456(torch.cat((O3,OC),dim=0))
        return O6


class D1(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.D1_1 = nn.Conv2d(in_channels =7, out_channels =64, kernel_size=4, stride=2, padding=1)
        self.D1_1_B = nn.BatchNorm2d(64)

        self.D1_2 = nn.Conv2d(in_channels =64, out_channels =128, kernel_size=4, stride=2, padding=1)
        self.D1_2_B = nn.BatchNorm2d(128)

        self.D1_3 = nn.Conv2d(in_channels =128, out_channels =256, kernel_size=4, stride=2, padding=1)
        self.D1_3_B = nn.BatchNorm2d(256)

        self.D1_4 = nn.Conv2d(in_channels =256, out_channels =512, kernel_size=4, stride=2, padding=1)
        self.D1_4_B = nn.BatchNorm2d(512)

        self.D1AB = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ).to(device)

        self.D1_5 = nn.ConvTranspose2d(in_channels =640, out_channels =1024, kernel_size=4, stride=2, padding=1)
        self.D1_5_B = nn.BatchNorm2d(1024)

        self.D1_6 = nn.ConvTranspose2d(in_channels =1074, out_channels =1024, kernel_size=1, stride=1)
        self.D1_6_B = nn.BatchNorm2d(1024)

        self.D1_7 = nn.ConvTranspose2d(in_channels =1024, out_channels =1, kernel_size=4, stride=4)

class D2(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.D2_123 = nn.sequential(
            nn.Conv2d(in_channels =3, out_channels =64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels =64, out_channels =128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels =128, out_channels =256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        ).to(device)

        self.D2ABC = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        ).to(device)

        self.D2_45 = nn.sequential(
            nn.Conv2d(in_channels =512, out_channels =512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            

            nn.ConvTranspose2d(in_channels =512, out_channels =1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)
        ).to(device)

        self.D2_67 = nn.sequential(
            nn.ConvTranspose2d(in_channels =(1024+108), out_channels =1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            

            nn.ConvTranspose2d(in_channels =1024, out_channels =1, kernel_size=4, stride=4),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)
        ).to(device)
    def forward(self, x_generated_image,x_segmented_image, x_design_encoding):
        O3 = self.D2_123(x_generated_image)
        OC = self.D2_ABC(x_segmented_image)
        # O5 = self.D2_45(torch.cat((Flatten(O3),Flatten(OC)),0))
        O5 = self.D2_45(torch.cat(O3,OC),0))
        d_by_4 = x_design_encoding.repeat(1,4)
        O6 = self.D2_67(torch.cat((layer5,d_by_4),0))
        return O6


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

loss = nn.BCELoss()
print("Using device:", device)

generator_optim = torch.optim.Adam(generator.parameters(), 2e-4, betas=(0.5, 0.999))
discriminator_optim = torch.optim.Adam(discriminator.parameters(), 2e-4, betas=(0.5, 0.999))