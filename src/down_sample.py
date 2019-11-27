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

def get_segmented_image(seg_img):
    img_no_downsample = seg_img
    img_no_downsample_copy = copy.deepcopy(img_no_downsample)
    #Create a tensor 4D, one dimention for each label [0,3]
    img_7_lay_tensor = torch.zeros(7, 128, 128)


    for i in range(0,7):
        # first set the value when the index != label to a dummy value = 4
        img_no_downsample_copy[img_no_downsample_copy != i] = 8
        # set the value = 1 when the index = label 
        img_no_downsample_copy[img_no_downsample_copy == i] = 1
        # convert from numpty array to tensor
        img_no_downsample_copy = torch.from_numpy(img_no_downsample_copy)
        #put the tensor in the corresponding index
        img_7_lay_tensor[i, :, :] = img_no_downsample_copy
        # reset the img_no_downsample_copy in order to process the second label 
        img_no_downsample_copy = copy.deepcopy(img_no_downsample)

    #replace the dummy value in the tensor with 0
    img_7_lay_tensor[img_7_lay_tensor == 8] = 0

    # convert the tensot to double instead of unit8 (usigned integer)
    img_7_lay_tensor = img_7_lay_tensor.type(torch.DoubleTensor)
    img_7_lay_tensor = img_7_lay_tensor.resize(1,7,128,128)

    img_7_lay_tensor = img_7_lay_tensor.resize(7,128,128)
    img_7_lay_tensor = img_7_lay_tensor.permute(0,2,1)
        
    return img_7_lay_tensor



def get_downsampled_image(img):
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
    img_4_lay_tensor = torch.nn.functional.interpolate(img_4_lay_tensor, scale_factor=(0.0625, 0.0625),  mode='bicubic', align_corners=True)

    img_4_lay_tensor = img_4_lay_tensor.resize(4,8,8)
    img_4_lay_tensor = img_4_lay_tensor.permute(0,2,1)
        
    return img_4_lay_tensor


def plot_tensor_image(t_img):
    for i in range(len(t_img)):
        segmented_tensor = t_img[i,:,:].resize(8,8)
        plt.imshow(segmented_tensor)
        plt.show()


def plot_tensor_seg_image(t_img):
    for i in range(len(t_img)):
        segmented_tensor = t_img[i,:,:].resize(128,128)
        plt.imshow(segmented_tensor)
        plt.show()

def get_downsampled_batch(batchsize,batch):
    batch_down_sampled = torch.ones(batchsize, 4, 8,8)
    for i in range(batchsize):
        batch_down_sampled[i]=get_downsampled_image(batch[i])
    return batch_down_sampled

#print('start downsampling')
#img = next(iter(new_dict))
#t_img = get_segmented_image(img)

#dd = t_img[0][0]
#plot_tensor_seg_image(t_img)

#print('end downsampling')