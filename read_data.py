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
    data = list(f[a_group_key])
    # Real images three channels instad of 0-255 values for a pixel we have normalized values between [-1;1]
    ih_data = list(f['ih'])
    test_picture = np.array(ih_data[0])
    import copy
    normalized = copy.deepcopy(test_picture)

    # normalize the pixel values from [-1;1] to [0;1]
    for channel in range(len(normalized)):
        normalized[channel] = (normalized[channel] - normalized[channel].min()) / (normalized[channel].max()-normalized[channel].min())
    
    test_picture_reshaped = torch.from_numpy(normalized)
    save_image(test_picture_reshaped,'img1.png')
    
    test_picture_reshaped = test_picture_reshaped.permute(1,2,0)
    print("After changing the shape")
    
    plt.imshow(test_picture_reshaped, interpolation='nearest')
    plt.show()
    # Mean for each channel 3x 128 x 128
    ih_mean_data = list(f['ih_mean'])
    first_img = data[70000][0]
    plt.imshow(first_img)
    plt.show()
    print("Hi")