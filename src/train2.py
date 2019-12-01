import data_loader
from torch.utils.data import DataLoader
import os
from matplotlib import pyplot as plt
import torch
import GANs
import down_sample
from torch.autograd import Variable
import pickle
import numpy as np
from IPython.display import display, clear_output


cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
X, y = None,None

# release
# if os.path.isfile(os.path.join(os.path.dirname(__file__),'..','data','data.pkl')):
#     with open(os.path.join(os.path.dirname(__file__),'..','data','data.pkl')) as handle:
#         X,y = pickle.load(handle)

# debug
loaded_data = None
# if os.path.isfile(os.path.join(os.path.dirname(__file__),'..','data','debug_data_10k.pkl')):
#     with open(os.path.join(os.path.dirname(__file__),'..','data','debug_data_10k.pkl'),'rb') as handle:
if os.path.isfile(os.path.join(os.path.dirname(__file__),'..','data','debug_data.pkl')):
    with open(os.path.join(os.path.dirname(__file__),'..','data','debug_data.pkl'),'rb') as handle:
        loaded_data = pickle.load(handle)
        X,y = loaded_data[0],loaded_data[1]
else:
    X, y = data_loader.load_data()
training_data = data_loader.FashionData(X,y,'train')
testing_data = data_loader.FashionData(X,y,'test')

batch_size = 50

train_loader = DataLoader(training_data, batch_size=batch_size,pin_memory=cuda)
test_loader  = DataLoader(testing_data, batch_size=batch_size, pin_memory=cuda)

# flatten_image_size = 128*128

# latent_dim_g2 = flatten_image_size + GANs.gausian_noise_size + GANs.human_attributes_size # 16492

G2 = GANs.Generator2()
D2 = GANs.Discriminator2()
if cuda:
    G2.cuda()
    D2.cuda()

loss_pix = torch.nn.L1Loss()
loss_GAN2 = torch.nn.BCELoss()
print("Using device:", device)

generator_2_optim = torch.optim.Adam(G2.parameters(), 0.0002, betas=(0.5, 0.999))
discriminator_2_optim = torch.optim.Adam(D2.parameters(), 0.0002, betas=(0.5, 0.999))



tmp_img = "tmp_gan_out.png"
discriminator_2_loss, generator_2_loss = [], []

num_epochs = 1
for epoch in range(num_epochs):
    batch_d_loss, batch_g_loss = [], []
    
    for i , data in enumerate(train_loader, 0):
        #need: I0:original image
        d, mS0, S0, label = data
        
        true_label = torch.ones(batch_size, 1,1,1).to(device)
        fake_label = torch.zeros(batch_size, 1,1,1).to(device)
        
        D2.zero_grad()
        

        
        
        #################### Update D #############################
        # loss 1. real image + real condition -> 1
        x_true_S0 = Variable(S0).to(device,dtype=torch.float)
        x_true_I0 = Variable(label).to(device,dtype=torch.float)
        x_true_d = Variable(d).to(device,dtype=torch.float)        
        output = D2.forward(x_true_d,x_true_I0,x_true_S0)
        
        error_true = loss_GAN2(output, true_label) 
        error_true.backward()

        # loss 2. sampled real image + wrong condition -> 0
        # shuffle d     
        # shuffle the true d row wise
        x_notmatch_d = x_true_d[torch.randperm(x_true_d.size()[0])]
        
        x_notmatch_d = Variable(x_notmatch_d).to(device)    
        output = D2.forward(x_notmatch_d ,x_true_I0,x_true_S0)

        error_notmatch = 0.2*loss_GAN2(output, fake_label) 
        error_notmatch.backward()

        # loss 3. generated fake image + real condition -> 0s
        # z = torch.randn(batch_size, 100, 1, 1,dtype=torch.float64)
        z = torch.randn(batch_size, 100,dtype=torch.float64)
        dz = torch.cat([d, z] , dim=1)
        dz = dz.view((batch_size,dz.shape[1],1,1))
        dz = Variable(dz).to(device,dtype=torch.float)
        x_g_S0 = Variable(S0).to(device,dtype=torch.float)
        
        I_tilde = G2.forward(dz,x_g_S0)

        x_fake_I = I_tilde
        # x_fake_S = Variable(S_tilde).to(device)
        x_fake_d = Variable(d).to(device,dtype=torch.float) 
        output = D2.forward(x_fake_d.detach(),x_fake_I.detach(),x_g_S0.detach())
        

        error_fake = 0.8*loss_GAN2(output, fake_label)#log(1-log(g(z)))
        error_fake.backward()
        discriminator_2_optim.step()

            
        G2.zero_grad()
        
        #################### Update G #############################
        
        
        # Step 4. Send fake data through discriminator _again_
        #         propagate the error of the generator and
        #         update G weights.
        output = D2.forward(x_fake_d,x_fake_I,x_g_S0)
        target = Variable(label).to(device,dtype=torch.float)
        error_D = loss_GAN2(output, true_label)
        # error_D.backward()
        error_seg = 50*loss_pix(x_fake_I,target)
        # error_seg.backward
        error_2_generator = error_seg + error_D
        error_2_generator.backward()
        generator_2_optim.step()
        
        # batch_d_loss.append((error_true/(error_true + error_fake + error_notmatch)).item())
        batch_d_loss.append((error_true + error_fake + error_notmatch).item())
        batch_g_loss.append(error_2_generator.item())

    discriminator_2_loss.append(np.mean(batch_d_loss))
    generator_2_loss.append(np.mean(batch_g_loss))
    
##################################
    print('Training epoch %d: discriminator_2_loss = %.5f, generator_2_loss = %.5f' % (epoch, discriminator_2_loss[epoch].item(), generator_2_loss[epoch].item()))


    # Generate data

    with torch.no_grad():
        zsample = torch.randn(100,dtype=torch.float64)
        dsample = []
        dsample.append(float(X['train']['gender'][0]))
        dsample.extend(data_loader.binary_representaiton(X['train']['color'][0],5))
        dsample.extend(data_loader.binary_representaiton(X['train']['sleeve'][0],3))
        dsample.extend(data_loader.binary_representaiton(X['train']['cate_new'][0],5))
        dsample.append(X['train']['r'][0])
        dsample.append(X['train']['g'][0])
        dsample.append(X['train']['b'][0])
        dsample.append(X['train']['y'][0])
        dsample.extend(X['train']['encoding'][0])
        dsample = np.array(dsample)
        dsample = torch.from_numpy(dsample)
        dzsample = torch.cat([dsample,zsample] , dim=0)
        dzsample = dzsample.view((1,dzsample.shape[0],1,1))
        dzsample = Variable(dzsample).to(device,dtype=torch.float)
        S0_sample = X['train']['segmented_image'][0]
        S0_sample = S0_sample.view((1,7,128,128))
        S0_sample = Variable(S0_sample).to(device,dtype=torch.float)
        I_tilde_sample = G2.forward(dzsample,S0_sample)
    


plt.plot(range(num_epochs), discriminator_2_loss)
plt.show()
plt.plot(range(num_epochs), generator_2_loss)
plt.show()

I_tilde_sample = I_tilde_sample.data.cpu().numpy()
I_tilde_sample =  I_tilde_sample.reshape(128,128,3)
fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(I_tilde_sample)

I_0_sample = label[0].data.cpu().numpy()
I_0_sample =I_0_sample.reshape(128,128,3)
ax= fig.add_subplot(122)
ax.imshow(I_0_sample[1])
plt.show()

#######save_image#########


# plt.savefig(tmp_img)
# plt.close(f)
# display(Image(filename=tmp_img))
# clear_output(wait=True)

# os.remove(tmp_img)