import h5py
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import torch

from torchvision.utils import save_image


filename = 'G2.h5'

ind = scipy.io.loadmat("ind.mat")
print("Coming in ind")
for key in ind.keys():
    print(key)

lang_org = scipy.io.loadmat("language_original.mat")
print("\nLanguage_original")
for key in lang_org.keys():
    print(key)

print("")
with h5py.File(filename, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    # Segmentated images 1x 128x 128 values from 0 to 6
    segmented_data = list(f['b_'])
    # Real images three channels instad of 0-255 values for a pixel we have normalized values between [-1;1]
    real_images = list(f['ih'])
    # test_picture = np.array(real_images[70000])
    import copy
    # normalized = copy.deepcopy(test_picture)

    # # normalize the pixel values from [-1;1] to [0;1]
    # for channel in range(len(normalized)):
    #     normalized[channel] = (normalized[channel] - normalized[channel].min()) / (normalized[channel].max()-normalized[channel].min())
    
    # test_picture_reshaped = torch.from_numpy(normalized)
    # save_image(test_picture_reshaped,'img1.png')

    for i in range(0,50):
        normalized = copy.deepcopy(np.array(real_images[i]))
        for channel in range(len(normalized)):
            normalized[channel] = (normalized[channel] - normalized[channel].min()) / (normalized[channel].max()-normalized[channel].min())
        test_picture_reshaped = torch.from_numpy(normalized)
        test_picture_reshaped = test_picture_reshaped.permute(0,2,1)
        image_name = 'images\img'+str(i)+'.png'
        save_image(test_picture_reshaped,image_name)

        segmented_image = np.array(segmented_data[i][0])
        segmented_tensor = torch.from_numpy(segmented_image)
        segmented_tensor = segmented_tensor.permute(1,0)
        segmented_image_name = 'images\seg'+str(i)+'.png'
        plt.imshow(segmented_tensor)
        plt.savefig(segmented_image_name)
        



    
    test_picture_reshaped = test_picture_reshaped.permute(1,2,0)
    print("After changing the shape")
    
    # plt.imshow(test_picture_reshaped, interpolation='nearest')
    # plt.show()
    # Mean for each channel 3x 128 x 128
    ih_mean_data = list(f['ih_mean'])
    # first_img = segmented_data[70000][0]
    # plt.imshow(first_img)
    # plt.show()
    print("Hi")