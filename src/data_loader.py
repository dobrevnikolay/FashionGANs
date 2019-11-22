

import os.path
import h5py
import scipy.io
import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np
from skimage import color

language_original_path = os.path.join(os.path.dirname(__file__),'../data/language_original.mat')
indeces_path = os.path.join(os.path.dirname(__file__),'../data/ind.mat')
segmented_images_raw_path = os.path.join(os.path.dirname(__file__),'../data/segmented_images.p')
real_images_raw_path = os.path.join(os.path.dirname(__file__),'../data/real_images.p')
h5_file_path = os.path.join(os.path.dirname(__file__),'../data/G2.h5')

# should we normalize the real_images
def construct_data(segmented_images,real_images,indeces,language):
    X = {}
    y = {}

    X['train'] = {}
    X['train']['gender'] =[]
    X['train']['color'] =[]
    X['train']['sleeve'] =[]
    X['train']['cate_new'] =[]
    X['train']['segmented_image'] = []
    X['train']['description'] = []
    X['train']['codeJ']
    X['train']['r'] = []
    X['train']['g'] = []
    X['train']['b'] = []
    X['train']['y'] = []

    X['test'] = {}
    X['test']['gender'] =[]
    X['test']['color'] =[]
    X['test']['sleeve'] =[]
    X['test']['cate_new'] =[]
    X['test']['segmented_image'] = []
    X['test']['description'] = []
    X['test']['codeJ'] =[]
    X['test']['r'] = []
    X['test']['g'] = []
    X['test']['b'] = []
    X['test']['y'] = []

    y['train'] = []
    y['test'] = []

    for i in range(len(indeces['train_ind'])):
        idx = indeces['train_ind'][i][0] - 1
        X['train']['gender'].append(language['gender_'][idx][0])
        X['train']['color'].append(language['color_'][idx][0])
        X['train']['sleeve'].append(language['sleeve_'][idx][0])
        X['train']['cate_new'].append(language['cate_new'][idx][0])
        X['train']['description'].append(str(language['engJ'][idx][0][0]))
        X['train']['segmented_image'].append(segmented_images[idx][0])   
        X['train']['codeJ'].append(str(language['codeJ'][idx][0][0]))

        r,g,b = np.median(real_images[idx][0]), np.median(real_images[idx][1]), np.median(real_images[idx][2])

        X['train']['r'].append(r)
        X['train']['g'].append(g)
        X['train']['b'].append(b)
        X['train']['y'].append(0.2125*r + 0.7154*g +  0.0721*b)

        y['train'].append(real_images[idx])

    for i in range(len(indeces['test_ind'])):
        idx = indeces['test_ind'][i][0] - 1
        X['test']['gender'].append(language['gender_'][idx][0])
        X['test']['color'].append(language['color_'][idx][0])
        X['test']['sleeve'].append(language['sleeve_'][idx][0])
        X['test']['cate_new'].append(language['cate_new'][idx][0])
        X['test']['description'].append(str(language['engJ'][idx][0][0]))
        X['test']['segmented_image'].append(segmented_images[idx][0])
        X['test']['codeJ'].append(str(language['codeJ'][idx][0][0]))

        r,g,b = np.median(real_images[idx][0]), np.median(real_images[idx][1]), np.median(real_images[idx][2])

        X['test']['r'].append(r)
        X['test']['g'].append(g)
        X['test']['b'].append(b)
        X['test']['y'].append(0.2125*r + 0.7154*g +  0.0721*b)

        y['test'].append(real_images[idx])
    
    return (X,y)

    
def normalize_pictures(real_images):
    for image in real_images:
        for channel in range(len(image)):
            image[channel] = (image[channel] - image[channel].min()) / (image[channel].max()-image[channel].min())
    return real_images


segmented_images = None
real_images = None

# check if the serialized images are present if not create them
if not(os.path.isfile(segmented_images_raw_path) and os.path.isfile(real_images_raw_path)):
    with h5py.File(h5_file_path, 'r') as f:   

        # Get the data
        # Segmentated images 1x 128x 128 values from 0 to 6
        segmented_images = list(f['b_'])
        # Real images three channels instad of 0-255 values for a pixel we have normalized values between [-1;1]
        import pickle
        pickle.dump(segmented_images, open(segmented_images_raw_path, 'wb')) 
        real_images = list(f['ih'])
        #normalize the real images
        real_images = normalize_pictures(real_images)
        pickle.dump(real_images, open(real_images_raw_path, 'wb')) 

if None == segmented_images:
    segmented_images = pickle.load(open(segmented_images_raw_path, 'rb'))

if None == real_images:
    real_images = pickle.load(open(real_images_raw_path,'rb'))

# plt.imshow(torch.from_numpy(real_images[0]).permute(2,1,0))

img = color.rgb2gray(torch.from_numpy(real_images[257]).permute(2,1,0))
median = np.median(img)
plt.imshow(img,cmap='gray')
plt.show()

# now read language
lang_org = scipy.io.loadmat(language_original_path)

# read the indeces as well
indeces = scipy.io.loadmat(indeces_path)

(X,y) = construct_data(segmented_images,real_images,indeces,lang_org)

# median value for each channel RGB where is Y channel grey scale



