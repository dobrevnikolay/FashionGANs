from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

image = Image.open('img1.png')
x = TF.to_tensor(image)

print(x.shape)

x = x.permute(2,1,0)

plt.imshow(x)
plt.show()

print("Hi")
