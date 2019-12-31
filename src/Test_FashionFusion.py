######### S0 -> G2 -> I_tilde ###################
def fusion_generation(imageind,dind,noise_size):
  with torch.no_grad():
    zsample = torch.randn(noise_size,dtype=torch.float64)
    dsample = []
    dsample.append(float(X['train']['gender'][dind]))
    dsample.extend(binary_representaiton(X['train']['color'][dind],5))
    dsample.extend(binary_representaiton(X['train']['sleeve'][dind],3))
    dsample.extend(binary_representaiton(X['train']['cate_new'][dind],5))
    dsample.append(X['train']['r'][dind])
    dsample.append(X['train']['g'][dind])
    dsample.append(X['train']['b'][dind])
    dsample.append(X['train']['y'][dind])
    dsample.extend(X['train']['encoding'][dind])
    dsample = np.array(dsample)
    dsample = torch.from_numpy(dsample)
    dzsample = torch.cat([dsample,zsample] , dim=0)
    dzsample = dzsample.view((1,dzsample.shape[0],1,1))
    dzsample = Variable(dzsample).to(device,dtype=torch.float)
    S0_sample = get_segmented_image_7(X['train']['segmented_image'][imageind])
    S0_sample = Variable(S0_sample).to(device,dtype=torch.float)
    S0_sample = S0_sample.view((1,7,128,128))
    I_tilde_sample = G2.forward(dzsample,S0_sample)
    I_tilde_sample = I_tilde_sample.data.cpu().numpy().T
    I_tilde_sample =  I_tilde_sample.reshape(128,128,3)
    I_0_sample = y['train'][imageind]
    I_0_sample = torch.from_numpy(I_0_sample)
    I_0_sample = I_0_sample.permute(2,1,0).numpy()

    mask = np.zeros((128,128,3))
    mask_inv = np.zeros((128,128,3))
    mask_1 = (S0_sample[0][3] + S0_sample[0][6]).cpu().numpy()
    mask_inv_1 = (S0_sample[0][0]+S0_sample[0][1]+S0_sample[0][2]+S0_sample[0][5]+S0_sample[0][4]).cpu().numpy()
    for ch in range(3):
      mask[:,:,ch] = mask_1
      mask_inv[:,:,ch] = mask_inv_1
    cloth = I_tilde_sample * mask
    fusion_image = I_0_sample*mask_inv+cloth 
  return(fusion_image,I_tilde_sample,I_0_sample)

def fusion_generation_test(imageind,dind,noise_size):
  with torch.no_grad():
    zsample = torch.randn(noise_size,dtype=torch.float64)
    dsample = []
    dsample.append(float(X['test']['gender'][dind]))
    dsample.extend(binary_representaiton(X['test']['color'][dind],5))
    dsample.extend(binary_representaiton(X['test']['sleeve'][dind],3))
    dsample.extend(binary_representaiton(X['test']['cate_new'][dind],5))
    dsample.append(X['test']['r'][dind])
    dsample.append(X['test']['g'][dind])
    dsample.append(X['test']['b'][dind])
    dsample.append(X['test']['y'][dind])
    dsample.extend(X['test']['encoding'][dind])
    dsample = np.array(dsample)
    dsample = torch.from_numpy(dsample)
    dzsample = torch.cat([dsample,zsample] , dim=0)
    dzsample = dzsample.view((1,dzsample.shape[0],1,1))
    dzsample = Variable(dzsample).to(device,dtype=torch.float)
    S0_sample = get_segmented_image_7(X['test']['segmented_image'][imageind])
    S0_sample = Variable(S0_sample).to(device,dtype=torch.float)
    S0_sample = S0_sample.view((1,7,128,128))
    I_tilde_sample = G2.forward(dzsample,S0_sample)
    I_tilde_sample = I_tilde_sample.data.cpu().numpy().T
    I_tilde_sample =  I_tilde_sample.reshape(128,128,3)
    I_0_sample = y['test'][imageind]
    I_0_sample = torch.from_numpy(I_0_sample)
    I_0_sample = I_0_sample.permute(2,1,0).numpy()

    mask = np.zeros((128,128,3))
    mask_inv = np.zeros((128,128,3))
    mask_1 = (S0_sample[0][3] + S0_sample[0][6]).cpu().numpy()
    mask_inv_1 = (S0_sample[0][0]+S0_sample[0][1]+S0_sample[0][2]+S0_sample[0][5]+S0_sample[0][4]).cpu().numpy()
    for ch in range(3):
      mask[:,:,ch] = mask_1
      mask_inv[:,:,ch] = mask_inv_1
    cloth = I_tilde_sample * mask
    fusion_image = I_0_sample*mask_inv+cloth 
  return(fusion_image,I_tilde_sample,I_0_sample)

######### 2GAN: G1 -> S_tilde -> G2 -> I_tilde ###################

def fusion_generation_2GAN(imageind,dind,noise_size):
  with torch.no_grad():
    zsample = torch.randn(noise_size,dtype=torch.float64)
    dsample = []
    dsample.append(float(X['train']['gender'][dind]))
    dsample.extend(binary_representaiton(X['train']['color'][dind],5))
    dsample.extend(binary_representaiton(X['train']['sleeve'][dind],3))
    dsample.extend(binary_representaiton(X['train']['cate_new'][dind],5))
    dsample.append(X['train']['r'][dind])
    dsample.append(X['train']['g'][dind])
    dsample.append(X['train']['b'][dind])
    dsample.append(X['train']['y'][dind])
    dsample.extend(X['train']['encoding'][dind])
    dsample = np.array(dsample)
    dsample = torch.from_numpy(dsample)
    dzsample = torch.cat([dsample,zsample] , dim=0)
    dzsample = dzsample.view((1,dzsample.shape[0],1,1))
    dzsample = Variable(dzsample).to(device,dtype=torch.float)
    mS0_sample = X['train']['down_sampled_images'][imageind]
    mS0_sample = mS0_sample.view((1,4,8,8))
    mS0_sample = Variable(mS0_sample).to(device,dtype=torch.float)
    S_tilde_sample = torch.exp(G1.forward(dzsample,mS0_sample))
    S0_sample = S_tilde_sample
    S0_sample = S0_sample.view((1,7,128,128))
    S0 = make_segmentation(S0_sample)
    S0 = torch.from_numpy(S0)
    S0 = Variable(S0).to(device,dtype=torch.float)
    I_tilde_sample = G2.forward(dzsample,S0)
    I_tilde_sample = I_tilde_sample.data.cpu().numpy().T
    I_tilde_sample =  I_tilde_sample.reshape(128,128,3)
    I_0_sample = y['train'][imageind]
    I_0_sample = torch.from_numpy(I_0_sample)
    I_0_sample = I_0_sample.permute(2,1,0).numpy()

    mask = np.zeros((128,128,3))
    mask_inv = np.zeros((128,128,3))
    mask_1 = (S0[0][3] + S0[0][6]).cpu().numpy()
    mask_inv_1 = (S0[0][0]+S0[0][1]+S0[0][2]+S0[0][5]+S0[0][4]).cpu().numpy()
    for ch in range(3):
      mask[:,:,ch] = mask_1
      mask_inv[:,:,ch] = mask_inv_1
    cloth = I_tilde_sample * mask
    fusion_image = I_0_sample*mask_inv+cloth 
  return(fusion_image,I_tilde_sample,I_0_sample)

def fusion_generation_test_2GAN(imageind,dind,noise_size):
  with torch.no_grad():
    zsample = torch.randn(noise_size,dtype=torch.float64)
    dsample = []
    dsample.append(float(X['test']['gender'][dind]))
    dsample.extend(binary_representaiton(X['test']['color'][dind],5))
    dsample.extend(binary_representaiton(X['test']['sleeve'][dind],3))
    dsample.extend(binary_representaiton(X['test']['cate_new'][dind],5))
    dsample.append(X['test']['r'][dind])
    dsample.append(X['test']['g'][dind])
    dsample.append(X['test']['b'][dind])
    dsample.append(X['test']['y'][dind])
    dsample.extend(X['test']['encoding'][dind])
    dsample = np.array(dsample)
    dsample = torch.from_numpy(dsample)
    dzsample = torch.cat([dsample,zsample] , dim=0)
    dzsample = dzsample.view((1,dzsample.shape[0],1,1))
    dzsample = Variable(dzsample).to(device,dtype=torch.float)
    mS0_sample = X['test']['down_sampled_images'][imageind]
    mS0_sample = mS0_sample.view((1,4,8,8))
    mS0_sample = Variable(mS0_sample).to(device,dtype=torch.float)
    S_tilde_sample = torch.exp(G1.forward(dzsample,mS0_sample))
    S0_sample = S_tilde_sample
    S0_sample = Variable(S0_sample).to(device,dtype=torch.float)
    S0_sample = S0_sample.view((1,7,128,128))
    S0 = make_segmentation(S0_sample)
    S0 = torch.from_numpy(S0)
    S0 = Variable(S0).to(device,dtype=torch.float)
    I_tilde_sample = G2.forward(dzsample,S0)
    I_tilde_sample = I_tilde_sample.data.cpu().numpy().T
    I_tilde_sample =  I_tilde_sample.reshape(128,128,3)
    I_0_sample = y['test'][imageind]
    I_0_sample = torch.from_numpy(I_0_sample)
    I_0_sample = I_0_sample.permute(2,1,0).numpy()

    mask = np.zeros((128,128,3))
    mask_inv = np.zeros((128,128,3))
    mask_1 = (S0[0][3] + S0[0][6]).cpu().numpy()
    mask_inv_1 = (S0[0][0]+S0[0][1]+S0[0][2]+S0[0][5]+S0[0][4]).cpu().numpy()
    for ch in range(3):
      mask[:,:,ch] = mask_1
      mask_inv[:,:,ch] = mask_inv_1
    cloth = I_tilde_sample * mask
    fusion_image = I_0_sample*mask_inv+cloth 
  return(fusion_image,I_tilde_sample,I_0_sample)

#input: (1,7,128,128) for each pixel, assign the maximum in 7 channels as 1, others as 0
def make_segmentation(S_tilde):
  S_tilde_new = np.zeros((1,7,128,128))
  for i in range(128):
    for j in range(128):
      maxcnt = 0
      maxind = 0
      for ch in range(7):
        if S_tilde[0][ch][i][j].item() > maxcnt :
          maxcnt = S_tilde[0][ch][i][j].item()
          maxind = ch
      S_tilde_new[0][maxind][i][j] = 1
#   S_tilde_new = S_tilde.cpu().numpy() # only test GAN2
  return S_tilde_new
  
########### code of loading might need to be modified to run in local instead of colab ###################

from torch.autograd import Variable
import numpy as np
import os
from torch.utils.data import DataLoader
#how to load the model:
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

G1 = Generator1().to(device,dtype=torch.float)
G1.load_state_dict(torch.load('/content/drive/My Drive/G1.pth'))
G2 = Generator2().to(device,dtype=torch.float)
G2.load_state_dict(torch.load('/content/drive/My Drive/G2.pth'))


#######   load the data if you haven't  #######
X, y = None,None

loaded_data = None
# if os.path.isfile(os.path.join(os.path.dirname(__file__),'..','data','debug_data_10k.pkl')):
#     with open(os.path.join(os.path.dirname(__file__),'..','data','debug_data_10k.pkl'),'rb') as handle:

if True:
    with open("/content/drive/My Drive/FashionData/debug_data_50k/debug_data_50k.pkl",'rb') as handle:
        print("I pickle")
        loaded_data = pickle.load(handle)
        X,y = loaded_data[0],loaded_data[1]
else:
    # X, y = load_data()
    print("we don't have data in the drive")

# samples for the result matrix
list = [40,667,235,115,90]

for imageind in list: 
  for dind in list:
    fusion,I_tilde,I_0 = fusion_generation_test(imageind,dind,gausian_noise_size)
    fusion_,I_tilde_,I_0_ = fusion_generation_test(imageind,imageind,gausian_noise_size)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(fusion)
    ax.set_xticks([])
    ax.set_yticks([])
    print(imageind,X['test']['description'][dind])
    plt.show()   