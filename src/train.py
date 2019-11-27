import data_loader
# import language_encoder
from torch.utils.data import DataLoader
import os
from matplotlib import pyplot as plt
import torch
import GANs
import down_sample
from torch.autograd import Variable
import pickle


cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
X, y = None,None

if os.path.isfile(os.path.dirname(__file__)+'../data/data.pickle'):
    with open(os.path.dirname(__file__)+'../data/data.pickle') as handle:
        X,y = pickle.load(handle)
else:
    X, y = data_loader.load_data()
training_data = data_loader.FashionData(X,y,'train')
testing_data = data_loader.FashionData(X,y,'test')

batch_size = 64

train_loader = DataLoader(training_data, batch_size=batch_size,pin_memory=cuda)
test_loader  = DataLoader(testing_data, batch_size=batch_size, pin_memory=cuda)





flatten_image_size = 128*128

latent_dim_g2 = flatten_image_size + gausian_noise_size + human_attributes_size # 16492

G1 = GANs.Generator1()
D1 = GANs.Discriminator1()

loss = torch.nn.BCELoss()
print("Using device:", device)

generator_1_optim = torch.optim.Adam(G1.parameters(), 2e-4, betas=(0.5, 0.999))
discriminator_1_optim = torch.optim.Adam(D1.parameters(), 2e-4, betas=(0.5, 0.999))



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

num_epochs = 50
for epoch in range(num_epochs):
    batch_d_loss, batch_g_loss = [], []
    
    for i , data in enumerate(train_loader, 0):
        # X['d'],X['mS0'], X['S0'],y
        # encoding
        #d = data
        batch_size = 13
        d, mS0, S0, label = data
        #batch_size = d.size()[0]
        # we flatten d into one big vector
        d_temp=[]
        for i in range(len(d)):
           for j in range(len(d[i])):
              d_temp.append(d[i][j])

        d = d_temp
        
        #flatten the downsampled image
        mS0 = mS0.view(batch_size, 64, 4)
        # flatten the segmented image
        S0 = S0.view(batch_size, 16384, 7)  # 128 * 128 = 16384 
        # True data is given label 1, while fake data is given label 0
        true_label = torch.ones(batch_size, 1).to(device)
        fake_label = torch.zeros(batch_size, 1).to(device)
        
        D1.zero_grad()
        G1.zero_grad()

        
        
        #################### Update D #############################
        # loss 1. real image + real condition -> 1
        x_true_S0 = Variable(S0).to(device)
        x_true_mS0 = Variable(mS0).to(device)
        x_true_d = Variable(d).to(device)        
        output = D1.forward(x_true_S0,x_true_mS0,x_true_d)
        
        error_true = loss(output, true_label) 
        error_true.backward()

        # loss 2. sampled real image + wrong condition -> 0
        # shuffle d     
        # shuffle the true d row wise
        x_notmatch_d = x_true_d[torch.randperm(x_true_d.size()[0])]

        #we pass the same real image and segmented image and we shuffle the description to get wrong description
        #x_notmatch_S0 = Variable(S0).to(device)
        #x_notmatch_mS0 = Variable(mS0).to(device)
        
        x_notmatch_d = Variable(x_notmatch_d).to(device)    
        output = D1.forward(x_true_S0 ,x_true_mS0,x_notmatch_d)

        error_true = loss(output, fake_label) 
        error_true.backward()

        # loss 3. generated fake image + real condition -> 0
        z = torch.randn(batch_size, 100, 1, 1)
        dz = torch.cat([d, z] , dim=1)
        dz = Variable(dz).to(device)
        x_g_mS0 = Variable(mS0).to(device)

        S_tilde = G1.foraward(dz,x_g_mS0)

        x_fake_S = S_tilde
        # x_fake_S = Variable(S_tilde).to(device)
        mS_tilde = down_sample.get_downsampled_batch(batch_size, S_tilde)
        x_fake_mS = Variable(mS_tilde).to(device)
        x_fake_d = Variable(d).to(device) 
        output = D1.forward(x_fake_S.detach(),x_fake_mS.detach(),x_fake_d.detach())
        

        error_fake = loss(output, fake_label)#log(1-log(g(z)))
        error_fake.backward()
        discriminator_1_optim.step()

            
        
        
        #################### Update G #############################
        
        
        # Step 4. Send fake data through discriminator _again_
        #         propagate the error of the generator and
        #         update G weights.
        output = D1.forward(x_fake_S,x_fake_mS,x_fake_d)
        
        error_generator = loss(output, true_label)
        error_generator.backward()
        generator_1_optim.step()
        
        batch_d_loss.append((error_true/(error_true + error_fake)).item())
        batch_g_loss.append(error_generator.item())

    discriminator_loss.append(np.mean(batch_d_loss))
    generator_loss.append(np.mean(batch_g_loss))
    
    # -- Plotting --
    f, axarr = plt.subplots(1, 2, figsize=(18, 7))

    # Loss
    ax = axarr[0]
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    ax.plot(np.arange(epoch+1), discriminator_loss)
    ax.plot(np.arange(epoch+1), generator_loss, linestyle="--")
    ax.legend(['Discriminator', 'Generator'])
    
    # Latent space samples
    ax = axarr[1]
    ax.set_title('Samples from generator')
    ax.axis('off')

    rows, columns = 8, 8
    
    # Generate data
    with torch.no_grad():
        z = torch.randn(rows*columns, latent_dim, 1, 1)
        z = Variable(z, requires_grad=False).to(device)
        x_fake = generator(z)
    
    canvas = np.zeros((28*rows, columns*28))
    for i in range(rows):
        for j in range(columns):
            idx = i % columns + rows * j
            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = x_fake.data[idx].cpu().numpy()
    ax.imshow(canvas, cmap='gray')
    
    plt.savefig(tmp_img)
    plt.close(f)
    display(Image(filename=tmp_img))
    clear_output(wait=True)

    os.remove(tmp_img)