import data_loader
import language_encoder
from torch.utils.data import DataLoader
import torch
import GANs.py
import get_downsampled_batch from down_sample


cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

# train language_encoder

# get encoded values for all pictures
encoded_description = []

(X,y) = data_loader.load_data(encoded_description)

training_data = data_loader.FashionData(X,y,'train')
testing_data = data_loader.FashionData(X,y,'test')

batch_size = 64

train_loader = DataLoader(training_data, batch_size=batch_size,
                          sampler=stratified_sampler(dset_train.train_labels), pin_memory=cuda)
test_loader  = DataLoader(testing_data, batch_size=batch_size, 
                          sampler=stratified_sampler(dset_test.test_labels), pin_memory=cuda)





flatten_image_size = 128*128

latent_dim_g2 = flatten_image_size + gausian_noise_size + human_attributes_size # 16492

G1 = Generator1()
D1 = Discriminator1()

loss = nn.BCELoss()
print("Using device:", device)

generator_1_optim = torch.optim.Adam(generator.parameters(), 2e-4, betas=(0.5, 0.999))
discriminator_1_optim = torch.optim.Adam(discriminator.parameters(), 2e-4, betas=(0.5, 0.999))



# S0: segmented image
# mS0: downsampled segmented image
# d: designed encoding
# dz: {d,z} encoding and noise
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
    
    for x, _ in train_loader:
        batch_size = x.size(0)
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

        # loss 2. sampled wrong image + real condition -> 0
        x_notmatch_S0 = Variable(S0).to(device)
        x_notmatch_mS0 = Variable(mS0).to(device)
        x_notmatch_d = Variable(d).to(device)    
        output = D1.forward(x_notmatch_S0,x_notmatch_mS0,x_notmatch_d)

        error_true = loss(output, false_label) 
        error_true.backward()

        # loss 3. generated fake image + real condition -> 0
        z = torch.randn(batch_size, 100, 1, 1)
        dz = torch.cat([d, z] , dim=1)
        dz = Variable(dz).to(device)
        x_g_mS0 = Variable(mS0).to(device)

        S_tilde = G1.foraward(dz,x_g_mS0)

        x_fake_S = S_tilde
        # x_fake_S = Variable(S_tilde).to(device)
        mS_tilde = get_downsampled_batch(batch_size, S_tilde)
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