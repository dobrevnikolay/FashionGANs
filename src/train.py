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

train_loader = DataLoader(training_data, batch_size=batch_size,pin_memory=cuda)
test_loader  = DataLoader(testing_data, batch_size=batch_size, pin_memory=cuda)


G1 = Generator1()
D1 = Discriminator1()
if cuda:
    G1.cuda()
    D1.cuda()

loss_seg = torch.nn.NLLLoss()
loss = torch.nn.BCELoss()
print("Using device:", device)




num_epochs = 75
discriminator_loss, generator_loss, generator_seg_loss = [], [], []

# decay = lambda global_step : np.linspace(1,0,num_epochs)[global_step]
# scheduler_gen = LambdaLR(generator_1_optim, decay)
# scheduler_disc = LambdaLR(discriminator_1_optim, decay)

for epoch in range(num_epochs):
    batch_d_loss, batch_g_loss,batch_g_seg_loss = [], [], []

    lr = 0.0005
    
    # decay = lr/(1+(50000/batch_size))

    
    for i , data in enumerate(train_loader, 0):

      generator_1_optim = torch.optim.Adam(G1.parameters(), lr, betas=(0.5, 0.999))
      discriminator_1_optim = torch.optim.Adam(D1.parameters(), lr/30, betas=(0.5, 0.999))
      d, mS0, S0, label = data
        

      soft_label = torch.randn(1)/100
      true_label = (torch.ones(batch_size,1)-0.1-soft_label).to(device)
      fake_label = (torch.zeros(batch_size,1)+0.1+soft_label).to(device)
      true_label_G = torch.ones(batch_size,1).to(device)

      D1.zero_grad()    
      if(0 == i and 0 == epoch%3):

    
        #################### Update D #############################
        noise_mask = torch.randn(batch_size,7,128,128)/10#add some noise to the input of D
        # loss 1. real image + real condition -> 1
        x_true_S0 = Variable(S0+noise_mask).to(device,dtype=torch.float)
        x_true_mS0 = Variable(mS0).to(device,dtype=torch.float)
        x_true_d = Variable(d).to(device,dtype=torch.float)        
        output = D1.forward(x_true_S0,x_true_mS0,x_true_d)
        
        error_true = loss(output, true_label) 
        error_true.backward()

        # loss 2. sampled real image + wrong condition -> 0
        sample_mS0 = np.zeros((batch_size,4,8,8))
        sample_S0 = np.zeros((batch_size,7,128,128))
        for bs in range(batch_size): 
          rnd = np.random.randint(0,len(y['train']))
          sample_mS0[bs] = X['train']['down_sampled_images'][rnd]
          sample_S0[bs] = get_segmented_image_7(X['train']['segmented_image'][rnd])
        x_notmatch_mS0 = Variable(torch.from_numpy(sample_mS0)).to(device,dtype=torch.float)  
        x_notmatch_S0 = Variable(torch.from_numpy(sample_S0)+noise_mask).to(device,dtype=torch.float)  
        x_notmatch_d = Variable(d).to(device,dtype=torch.float)    
        output = D1.forward(x_notmatch_S0 ,x_notmatch_mS0,x_notmatch_d)

        error_notmatch = 0.3*loss(output, fake_label) 
        error_notmatch.backward()

        # loss 4. generated fake image + real condition -> 0
        z = torch.randn(batch_size, gausian_noise_size,dtype=torch.float64)
        dz = torch.cat([d, z] , dim=1)
        dz = dz.view((batch_size,dz.shape[1],1,1))
        dz = Variable(dz).to(device,dtype=torch.float)
        x_g_mS0 = Variable(mS0).to(device,dtype=torch.float)

        S_tilde = G1.forward(dz,x_g_mS0)

        x_fake_S = (torch.exp(S_tilde).cpu() + noise_mask).to(device,dtype=torch.float)
        # mS_tilde = get_segmented_image_7_s_tilde(batch_size, S_tilde)
        # x_fake_mS = get_downsampled_image_4_mS0(batch_size, mS_tilde)
        # x_fake_mS = Variable(x_fake_mS).to(device,dtype=torch.float)
        x_fake_d = Variable(d).to(device,dtype=torch.float) 
        output = D1.forward(x_fake_S.detach(),x_g_mS0.detach(),x_fake_d.detach())
        

        error_fake = 0.7*loss(output, fake_label)
        error_fake.backward()
        discriminator_1_optim.step()

      #################### Update G #############################
      
      G1.zero_grad()
      # Step 4. Send fake data through discriminator _again_
      #         propagate the error of the generator and
      #         update G weights.

      z = torch.randn(batch_size, gausian_noise_size,dtype=torch.float64)
      real_d = Variable(d).to(device,dtype=torch.float) 
      dz = torch.cat([d, z] , dim=1)
      dz = dz.view((batch_size,dz.shape[1],1,1))
      dz = Variable(dz).to(device,dtype=torch.float)
      x_g_mS0 = Variable(mS0).to(device,dtype=torch.float)

      S_tilde = G1.forward(dz,x_g_mS0)
      
      output = D1.forward(torch.exp(S_tilde),x_g_mS0,x_fake_d)
      target = Variable(get_segmented_image_1(batch_size,S0)).to(device,dtype=torch.long)
      error_D = loss(output, true_label_G)
      error_seg = loss_seg(S_tilde,target) # weak regularization
#       error_seg = 50*loss_seg(S_tilde,target) # strong regularization
      error_generator = error_seg + error_D
      error_generator.backward()
      generator_1_optim.step()
      
      batch_d_loss.append((error_true + error_fake + error_notmatch).item()/2)
      # batch_g_loss.append(error_generator.item())
      batch_g_loss.append(error_D.item())
      batch_g_seg_loss.append(error_seg.item())

      lr = lr * 0.9999 # 

    discriminator_loss.append(np.mean(batch_d_loss))
    generator_loss.append(np.mean(batch_g_loss))
    generator_seg_loss.append(np.mean(batch_g_seg_loss))

    print('Training epoch %d: discriminator_loss = %.5f, generator_loss = %.5f, generate_seg_loss=%.5f' % (epoch, discriminator_loss[epoch].item(), generator_loss[epoch].item(),generator_seg_loss[epoch].item()))


#save the model
torch.save(G1.state_dict(), '/content/drive/My Drive/G1.pth')
torch.save(D1.state_dict(), '/content/drive/My Drive/D1.pth')
    # Generate data
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
    mS0_sample = X['train']['down_sampled_images'][0]
    mS0_sample = mS0_sample.view((1,4,8,8))
    mS0_sample = Variable(mS0_sample).to(device,dtype=torch.float)
    S_tilde_sample = torch.exp(G1.forward(dzsample,mS0_sample))
    


fig1 = plt.figure(1)
plt.plot(range(num_epochs), discriminator_loss)
plt.xlabel('Epochs')  
plt.ylabel('Discriminator1_Loss')  
plt.show()
fig2 = plt.figure(2)
plt.plot(range(num_epochs), generator_loss)
plt.xlabel('Epochs')  
plt.ylabel('Generator1_Loss')  
plt.show()


S_tilde_sample = S_tilde_sample.data.cpu().numpy()

S_0_sample = X['train']['segmented_image'][0]#for old dataset
S_0_sample = get_segmented_image_7(X['train']['segmented_image'][0])#for new dataset


S_tilde_sample =  S_tilde_sample.reshape(7,128,128)
S_0_sample =S_0_sample.reshape(7,128,128)


for i in  range(len(S_tilde_sample)):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(S_tilde_sample[i])
    ax[0].set_title("Generated")
    ax[1].imshow(S_0_sample[i])
    ax[1].set_title("Original")
    plt.show()

# np.save("first_gan",S_tilde_sample)
# print("Saving the sample")
