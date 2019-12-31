# load the data
from torch.autograd import Variable
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
########### code of loading might need to be modified to run in local instead of colab ###################


cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
X, y = None,None

loaded_data = None
# if os.path.isfile(os.path.join(os.path.dirname(__file__),'..','data','debug_data_10k.pkl')):
#     with open(os.path.join(os.path.dirname(__file__),'..','data','debug_data_10k.pkl'),'rb') as handle:

if True:
    with open("/content/drive/My Drive/FashionData/debug_data_50k/debug_data_50k.pkl",'rb') as f:
        print("I pickle")
        loaded_data = pickle.load(f)
        # X,y = pickle.load(handle)
        X,y = loaded_data[0],loaded_data[1]
else:
    # X, y = load_data()
    print("we don't have data in the drive")
training_data = FashionData(X,y,'train')
testing_data = FashionData(X,y,'test')

batch_size = 10

train_loader = DataLoader(training_data, batch_size=batch_size,pin_memory=cuda,shuffle=True)
test_loader  = DataLoader(testing_data, batch_size=batch_size, pin_memory=cuda,shuffle=True)


print("Using device:", device)

G2 = Generator2()
D2 = Discriminator2()
if cuda:
    G2.cuda()
    D2.cuda()

loss_pix = torch.nn.L1Loss()
loss_GAN2 = torch.nn.BCELoss()
print("Using device:", device)



num_epochs = 75




discriminator_2_loss, generator_2_loss, generator_2_L1loss,discriminator_2_loss_true,discriminator_2_loss_fake,discriminator_2_loss_notmatch  = [], [], [],[],[],[]

for epoch in range(num_epochs):
    batch_d_loss, batch_g_loss, batch_g_L1loss,batch_d_loss_true,batch_d_loss_fake,batch_d_loss_notmatch = [], [], [],[],[],[]
    lr = 0.001

    for i , data in enumerate(train_loader, 0):
        generator_2_optim = torch.optim.Adam(G2.parameters(), lr, betas=(0.5, 0.999))
        discriminator_2_optim = torch.optim.Adam(D2.parameters(), lr/30, betas=(0.5, 0.999))
        
        
        #need: I0:original image
        d, mS0, S0, label = data
        
        soft_label = torch.randn(1,1,1)/100
        true_label = (torch.ones(batch_size, 1,1,1)-0.1-soft_label).to(device)
        fake_label = (torch.zeros(batch_size, 1,1,1)+0.1+soft_label).to(device)
        true_label_G = torch.ones(batch_size, 1,1,1).to(device)
        

        if(0 == i and 0 == epoch%2):

          D2.zero_grad()  
          
          #################### Update D #############################
          noise_mask = torch.randn(batch_size,3,128,128)/10#add some noise to the input of D
          # loss 1. real image + real condition -> 1
          x_true_S0 = Variable(S0).to(device,dtype=torch.float)
          x_true_I0 = Variable(label+noise_mask).to(device,dtype=torch.float)
          x_true_d = Variable(d).to(device,dtype=torch.float)        
          output = D2.forward(x_true_d,x_true_I0,x_true_S0)
          
          error_true = loss_GAN2(output, true_label) 
          # loss 2. sampled real image + wrong condition -> 0  
          sample_I = np.zeros((batch_size,3,128,128))
          sample_S0 = np.zeros((batch_size,7,128,128))
          for bs in range(batch_size):
            rnd = np.random.randint(0,len(y['train']))
            sample_I[bs] = y['train'][rnd]
            sample_S0[bs] = get_segmented_image_7(X['train']['segmented_image'][rnd])
          x_notmatch_I = Variable(torch.from_numpy(sample_I)+noise_mask).to(device,dtype=torch.float)
          x_notmatch_S0 = Variable(torch.from_numpy(sample_S0)).to(device,dtype=torch.float)  
          # x_notmatch_d = x_true_d[torch.randperm(x_true_d.size()[0])] # another way: shuffle d
          x_notmatch_d = Variable(d).to(device,dtype=torch.float)    
          # x_notmatch_d = Variable(x_notmatch_d).to(device)    
          output = D2.forward(x_notmatch_d ,x_true_I0,x_notmatch_S0)

          error_notmatch = 0.3*loss_GAN2(output, fake_label) 

          # loss 3. generated fake image + real condition -> 0s
          z = torch.randn(batch_size, gausian_noise_size,dtype=torch.float64)
          dz = torch.cat([d, z] , dim=1)
          dz = dz.view((batch_size,dz.shape[1],1,1))
          dz = Variable(dz).to(device,dtype=torch.float)
          x_true_S0_g = Variable(S0).to(device,dtype=torch.float)
          
          I_tilde = G2.forward(dz,x_true_S0_g)

          x_fake_I = (I_tilde.cpu()+noise_mask).to(device,dtype=torch.float)
          x_fake_d = Variable(d).to(device,dtype=torch.float) 
          output = D2.forward(x_fake_d,x_fake_I.detach(),x_true_S0_g)
          

          error_fake = 0.7*loss_GAN2(output, fake_label)
          err_D = error_true+error_fake+error_notmatch
          err_D.backward()
          discriminator_2_optim.step()

              
        
        #################### Update G #############################
        G2.zero_grad()

        
        # Step 4. Send fake data through discriminator _again_
        #         propagate the error of the generator and
        #         update G weights.
        x_fake_d = Variable(d).to(device,dtype=torch.float)
        z = torch.randn(batch_size, gausian_noise_size,dtype=torch.float64)
        dz = torch.cat([d, z] , dim=1)
        dz = dz.view((batch_size,dz.shape[1],1,1))
        dz = Variable(dz).to(device,dtype=torch.float)
        x_true_S0_g = Variable(S0).to(device,dtype=torch.float)
        
        I_tilde = G2.forward(dz,x_true_S0_g)

        x_fake_I = I_tilde
        
        
        output = D2.forward(x_fake_d,x_fake_I,x_true_S0_g)
        target = Variable(label).to(device,dtype=torch.float)
        error_D = loss_GAN2(output, true_label_G)
        # error_D.backward()
        error_pix = loss_pix(x_fake_I,target) # weak regularization
#         error_pix = 50*loss_pix(x_fake_I,target) # strong regularization
        # error_seg.backward
        error_2_generator = error_pix + error_D
        error_2_generator.backward()
        generator_2_optim.step()
        
        lr = lr * 0.9999
        
        # batch_d_loss.append((error_true/(error_true + error_fake + error_notmatch)).item())
        batch_d_loss.append(0.5*(error_true + error_fake + error_notmatch).item())
        batch_d_loss_true.append(error_true.item())
        batch_d_loss_fake.append(error_fake.item())
        batch_d_loss_notmatch.append(error_notmatch.item())
        batch_g_loss.append(error_D.item())
        batch_g_L1loss.append(error_pix.item())


#         if epoch<2:

#           if i%5000==0:

#             with torch.no_grad():
#                 zsample = torch.randn(gausian_noise_size,dtype=torch.float64)
#                 dsample = []
#                 dsample.append(float(X['train']['gender'][0]))
#                 dsample.extend(binary_representaiton(X['train']['color'][0],5))
#                 dsample.extend(binary_representaiton(X['train']['sleeve'][0],3))
#                 dsample.extend(binary_representaiton(X['train']['cate_new'][0],5))
#                 dsample.append(X['train']['r'][0])
#                 dsample.append(X['train']['g'][0])
#                 dsample.append(X['train']['b'][0])
#                 dsample.append(X['train']['y'][0])
#                 dsample.extend(X['train']['encoding'][0])
#                 dsample = np.array(dsample)
#                 dsample = torch.from_numpy(dsample)
#                 dzsample = torch.cat([dsample,zsample] , dim=0)
#                 dzsample = dzsample.view((1,dzsample.shape[0],1,1))
#                 dzsample = Variable(dzsample).to(device,dtype=torch.float)
#                 S0_sample = get_segmented_image_7(X['train']['segmented_image'][0])
#                 S0_sample = S0_sample.view((1,7,128,128))
#                 S0_sample = Variable(S0_sample).to(device,dtype=torch.float)
#                 I_tilde_sample = G2.forward(dzsample,S0_sample)

#             I_tilde_sample = I_tilde_sample.data.cpu().numpy().T
#             I_tilde_sample =  I_tilde_sample.reshape(128,128,3)
#             fig = plt.figure()
#             ax = fig.add_subplot(121)
#             ax.imshow(I_tilde_sample)
#             ax.set_title("Generated")

#             I_0_sample = y['train'][0]
#             I_0_sample = torch.from_numpy(I_0_sample)
#             I_0_sample = I_0_sample.permute(2,1,0)
#             ax= fig.add_subplot(122)
#             ax.imshow(I_0_sample)
#             ax.set_title("Original")
#             plt.savefig('/content/drive/My Drive/GAN2/i_{}.png'.format(epoch))
#             plt.show()

    discriminator_2_loss.append(np.mean(batch_d_loss))
    generator_2_loss.append(np.mean(batch_g_loss))
    generator_2_L1loss.append(np.mean(batch_g_L1loss))
    discriminator_2_loss_true.append(np.mean(batch_d_loss_true))
    discriminator_2_loss_notmatch.append(np.mean(batch_d_loss_notmatch))
    discriminator_2_loss_fake.append(np.mean(batch_d_loss_fake))
   
    print('Training epoch %d: discriminator_2_loss = %.5f, generator_2_loss = %.5f, generator_2_L1loss = %.5f' % (epoch, discriminator_2_loss[epoch].item(), generator_2_loss[epoch].item(),generator_2_L1loss[epoch].item()))
    print('Training epoch %d: d_true = %.5f, d_notmatch = %.5f, d_fake = %.5f' % (epoch, discriminator_2_loss_true[epoch].item(), discriminator_2_loss_notmatch[epoch].item(),discriminator_2_loss_fake[epoch].item()))

# save image per 5 epoch
    if True:
      with torch.no_grad():
          zsample = torch.randn(gausian_noise_size,dtype=torch.float64)
          dsample = []
          dsample.append(float(X['train']['gender'][0]))
          dsample.extend(binary_representaiton(X['train']['color'][0],5))
          dsample.extend(binary_representaiton(X['train']['sleeve'][0],3))
          dsample.extend(binary_representaiton(X['train']['cate_new'][0],5))
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
          S0_sample = get_segmented_image_7(X['train']['segmented_image'][0])
          S0_sample = S0_sample.view((1,7,128,128))
          S0_sample = Variable(S0_sample).to(device,dtype=torch.float)
          I_tilde_sample = G2.forward(dzsample,S0_sample)

      I_tilde_sample = I_tilde_sample.data.cpu().numpy().T
      I_tilde_sample =  I_tilde_sample.reshape(128,128,3)
      fig = plt.figure()
      ax = fig.add_subplot(121)
      ax.imshow(I_tilde_sample)
      ax.set_title("Generated")

      I_0_sample = y['train'][0]
      I_0_sample = torch.from_numpy(I_0_sample)
      I_0_sample = I_0_sample.permute(2,1,0)
      ax= fig.add_subplot(122)
      ax.imshow(I_0_sample)
      ax.set_title("Original")
      plt.savefig('/content/drive/My Drive/GAN2/epoch{}.png'.format(epoch))
      plt.show()

    # Generate data
# save model
torch.save(G2.state_dict(), '/content/drive/My Drive/G2.pth')
torch.save(D2.state_dict(), '/content/drive/My Drive/D2.pth')
with torch.no_grad():
    zsample = torch.randn(gausian_noise_size,dtype=torch.float64)
    dsample = []
    dsample.append(float(X['train']['gender'][0]))
    dsample.extend(binary_representaiton(X['train']['color'][0],5))
    dsample.extend(binary_representaiton(X['train']['sleeve'][0],3))
    dsample.extend(binary_representaiton(X['train']['cate_new'][0],5))
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
    S0_sample = get_segmented_image_7(X['train']['segmented_image'][0])
    S0_sample = S0_sample.view((1,7,128,128))
    S0_sample = Variable(S0_sample).to(device,dtype=torch.float)
    I_tilde_sample = G2.forward(dzsample,S0_sample)
    
fig1 = plt.figure(1)
plt.plot(range(num_epochs), discriminator_2_loss)
plt.xlabel('Epochs')  
plt.ylabel('Discriminator2_Loss')  
plt.show()
fig2 = plt.figure(2)
plt.plot(range(num_epochs), generator_2_loss)
plt.xlabel('Epochs')  
plt.ylabel('Generator2_Loss')  
plt.show()
fig3 = plt.figure(3)
plt.plot(range(num_epochs), generator_2_L1loss)
plt.xlabel('Epochs')  
plt.ylabel('Generator2_L1Loss') 
plt.show()


I_tilde_sample = I_tilde_sample.data.cpu().numpy().T
I_tilde_sample =  I_tilde_sample.reshape(128,128,3)

fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(I_tilde_sample)
ax.set_title("Generated")

I_0_sample = y['train'][0]
I_0_sample = torch.from_numpy(I_0_sample)
I_0_sample = I_0_sample.permute(2,1,0)
ax= fig.add_subplot(122)
ax.imshow(I_0_sample)
ax.set_title("Original")
plt.show()


    