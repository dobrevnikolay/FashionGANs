from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch



infile = open("serialized_images.p",'rb')
new_dict = pickle.load(infile)
infile.close()

img = next(iter(new_dict))
#img[img > 3] = 3
print(img.shape)


segmented_image = np.array(img)

segmented_tensor = torch.from_numpy(segmented_image)
segmented_tensor = segmented_tensor.permute(2,1,0)
segmented_tensor = segmented_tensor.resize(128,128)
print(segmented_tensor.shape)
#segmented_image_name = 'images\seg'+str(i)+'.png'

#mask = imresize([1,2,3,4], (128, 128), interp='bicubic').astype('float32')


print('before print')
plt.imshow(segmented_tensor)
plt.show()
