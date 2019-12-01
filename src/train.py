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

flatten_image_size = 128*128

latent_dim_g2 = flatten_image_size + GANs.gausian_noise_size + GANs.human_attributes_size # 16492

G1 = GANs.Generator1()
D1 = GANs.Discriminator1()
if cuda:
    G1.cuda()
    D1.cuda()

loss_seg = torch.nn.NLLLoss()
loss = torch.nn.BCELoss()
print("Using device:", device)

generator_1_optim = torch.optim.Adam(G1.parameters(), 0.0002, betas=(0.5, 0.999))
discriminator_1_optim = torch.optim.Adam(D1.parameters(), 0.0002, betas=(0.5, 0.999))



# S0: segmented image
# mS0: downsampled segmented image
# d: designed encoding
# dz: {d,z} encoding and noise
# y: real image
#TODO assign data to these variables above
#TODO whether create variables for each step or just once?
#TODO should batch works?
#TODO debug the language encoder function
#TODO encode all data or run encoder during training
#TODO whether need to add condition loss in updating G2?
tmp_img = "tmp_gan_out.png"
discriminator_loss, generator_loss = [], []

num_epochs = 20
for epoch in range(num_epochs):
    batch_d_loss, batch_g_loss = [], []
    
    for i , data in enumerate(train_loader, 0):
        
        d, mS0, S0, label = data
        
        true_label = torch.ones(batch_size, 1).to(device)
        fake_label = torch.zeros(batch_size, 1).to(device)
        
        D1.zero_grad()
        

        
        
        #################### Update D #############################
        # loss 1. real image + real condition -> 1
        x_true_S0 = Variable(S0).to(device,dtype=torch.float)
        x_true_mS0 = Variable(mS0).to(device,dtype=torch.float)
        x_true_d = Variable(d).to(device,dtype=torch.float)        
        output = D1.forward(x_true_S0,x_true_mS0,x_true_d)
        
        error_true = loss(output, true_label) 
        error_true.backward()

        # loss 2. sampled real image + wrong condition -> 0
        # shuffle d     
        # shuffle the true d row wise
        x_notmatch_d = x_true_d[torch.randperm(x_true_d.size()[0])]
        
        x_notmatch_d = Variable(x_notmatch_d).to(device)    
        output = D1.forward(x_true_S0 ,x_true_mS0,x_notmatch_d)

        error_notmatch = loss(output, fake_label) 
        error_notmatch.backward()

        # loss 3. generated fake image + real condition -> 0s
        # z = torch.randn(batch_size, 100, 1, 1,dtype=torch.float64)
        z = torch.randn(batch_size, 100,dtype=torch.float64)
        dz = torch.cat([d, z] , dim=1)
        dz = dz.view((batch_size,dz.shape[1],1,1))
        dz = Variable(dz).to(device,dtype=torch.float)
        x_g_mS0 = Variable(mS0).to(device,dtype=torch.float)

        S_tilde = G1.forward(dz,x_g_mS0)

        x_fake_S = S_tilde
        # x_fake_S = Variable(S_tilde).to(device)
        mS_tilde = down_sample.get_segmented_image_7_s_tilde(batch_size, S_tilde)
        x_fake_mS = down_sample.get_downsampled_image_4_mS0(batch_size, mS_tilde)
        x_fake_mS = Variable(x_fake_mS).to(device,dtype=torch.float)
        x_fake_d = Variable(d).to(device,dtype=torch.float) 
        output = D1.forward(x_fake_S.detach(),x_fake_mS.detach(),x_fake_d.detach())
        

        error_fake = loss(output, fake_label)#log(1-log(g(z)))
        error_fake.backward()
        discriminator_1_optim.step()

            
        G1.zero_grad()
        
        #################### Update G #############################
        
        
        # Step 4. Send fake data through discriminator _again_
        #         propagate the error of the generator and
        #         update G weights.
        output = D1.forward(x_fake_S,x_fake_mS,x_fake_d)
        target = Variable(down_sample.get_segmented_image_1(batch_size,S0)).to(device,dtype=torch.long)
        error_D = loss(output, true_label)
        # error_D.backward()
        error_seg = 50*loss_seg(x_fake_S,target)
        # error_seg.backward
        error_generator = error_seg + error_D
        error_generator.backward()
        generator_1_optim.step()
        
        # batch_d_loss.append((error_true/(error_true + error_fake + error_notmatch)).item())
        batch_d_loss.append((error_true + error_fake + error_notmatch).item())
        batch_g_loss.append(error_generator.item())

    discriminator_loss.append(np.mean(batch_d_loss))
    generator_loss.append(np.mean(batch_g_loss))
    
##################################
    print('Training epoch %d: discriminator_loss = %.5f, generator_loss = %.5f' % (epoch, discriminator_loss[epoch].item(), generator_loss[epoch].item()))
   
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
        mS0_sample = X['train']['down_sampled_images'][0]
        mS0_sample = mS0_sample.view((1,4,8,8))
        mS0_sample = Variable(mS0_sample).to(device,dtype=torch.float)
        S_tilde_sample = G1.forward(dzsample,mS0_sample)
    


plt.plot(range(num_epochs), discriminator_loss)
plt.show()
plt.plot(range(num_epochs), generator_loss)
plt.show()

S_tilde_sample = S_tilde_sample.data.cpu().numpy()
# S_0_sample = X['train']['segmented_image'][0].data.cpu().numpy()
# S_0_sample = down_sample.get_segmented_image_7_s_tilde(X['train']['segmented_image'][0])
S_0_sample = X['train']['segmented_image'][0]

S_tilde_sample =  S_tilde_sample.reshape(7,128,128)
S_0_sample =S_0_sample.reshape(7,128,128)
fig, ax = plt.subplots(nrows=7, ncols=2)#,figsize=(15,15))
for row_index, row in enumerate(ax,0):
    row[0].imshow(S_tilde_sample[row_index])
    row[1].imshow(S_0_sample[row_index])

plt.show()

#######save_image#########


# plt.savefig(tmp_img)
# plt.close(f)
# display(Image(filename=tmp_img))
# clear_output(wait=True)

# os.remove(tmp_img)