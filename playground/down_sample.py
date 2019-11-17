from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import copy
import os.path



# Load segmented images
segmented_images_raw_path = os.path.join(os.path.dirname(__file__),'../data/segmented_images.p')
infile = open(segmented_images_raw_path,'rb')
new_dict = pickle.load(infile)
# Close the stream
infile.close()

# Get next element from the dictionay
img = next(iter(new_dict))
img_no_downsample = img
# L´ : only use the first four lables. if label > 3, set label = 3
img_no_downsample[img_no_downsample > 3] = 3
# copy
img_no_downsample_copy = copy.deepcopy(img_no_downsample)

#Create a tensor 4D, one dimention for each label [0,3]
img_4_lay_tensor = torch.zeros(4, 128, 128)


for i in range(0,4):
    # first set the value when the index != label to a dummy value = 4
    img_no_downsample_copy[img_no_downsample_copy != i] = 4
    # set the value = 1 when the index = label 
    img_no_downsample_copy[img_no_downsample_copy == i] = 1
    # convert from numpty array to tensor
    img_no_downsample_copy = torch.from_numpy(img_no_downsample_copy)
    #put the tensor in the corresponding index
    img_4_lay_tensor[i, :, :] = img_no_downsample_copy
    # reset the img_no_downsample_copy in order to process the second label 
    img_no_downsample_copy = copy.deepcopy(img_no_downsample)

#replace the dummy value in the tensor with 0
img_4_lay_tensor[img_4_lay_tensor == 4] = 0

# convert the tensot to double instead of unit8 (usigned integer)
img_4_lay_tensor = img_4_lay_tensor.type(torch.DoubleTensor)
img_4_lay_tensor = img_4_lay_tensor.resize(1,4,128,128)
#segmented_image = segmented_image.view(1,-1,-1,-1)
# downsampling by 1/8
img_4_lay_tensor = torch.nn.functional.interpolate(img_4_lay_tensor, scale_factor=(0.125, 0.125),  mode='bicubic', align_corners=True)

img_4_lay_tensor = img_4_lay_tensor.resize(4,16,16)
img_4_lay_tensor = img_4_lay_tensor.permute(0,2,1)
for i in range(len(img_4_lay_tensor)):
    segmented_tensor = img_4_lay_tensor[i,:,:].resize(16,16)
    plt.imshow(segmented_tensor)
    plt.show()
