#################Language Encoder Loading mat File###################
import sys
from random import randint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

############### parameters ###################
dim_voc = 539 # size of vocabulary
num_layers = 2 
bsz = 1 #batch only 1 for convenience
dim_h = 100 # rnn hidden units
dim_cate_new = 19 # category of clothes
dim_color = 17 #color of clothes
dim_gender = 2 # gender
dim_sleeve = 4 # length of sleeve

m = 78979 # size of test set
m_train = 60000 # size of training set
m_test = 70000


############### loading ###################


mat = loadmat('src/language_encoder/language_original.mat')
for k, v in mat.items():
    exec(k +  " = mat['" + k + "']")



indmat = loadmat('src/language_encoder/ind.mat')
train_ind = torch.IntTensor(m_train)
for i in range(m_train):
    train_ind[i] = int(indmat['train_ind'][i][0] - 1)

data_cate_new = torch.IntTensor(m, 1)
data_color = torch.IntTensor(m, 1)
data_gender = torch.IntTensor(m, 1)
data_sleeve = torch.IntTensor(m, 1)
for i in range(m):
    data_cate_new[i][0] = int(cate_new[i][0] - 1)
    data_color[i][0] = int(color_[i][0] - 1)
    data_gender[i][0] = int(gender_[i][0])
    data_sleeve[i][0] = int(sleeve_[i][0] - 1)

#################CUDA###################
use_cuda = torch.cuda.is_available()
# use_cuda = 0
#haven't test with CUDA. Set it as 0 to use CPU if you've installed CUDA 

print("Running GPU.") if use_cuda else print("No GPU available.")


def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x


def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()

############### network ###################

class language_encoder(nn.Module):
    def __init__(self):
        super(language_encoder, self).__init__()
        self.rnn = nn.RNN(dim_voc, dim_h, num_layers)
        self.net_cate_new = nn.Linear(dim_h, dim_cate_new)
        self.net_color = nn.Linear(dim_h, dim_color)
        self.net_gender = nn.Linear(dim_h, dim_gender)
        self.net_sleeve = nn.Linear(dim_h, dim_sleeve)

    def forward(self, x):
        h0 = get_variable(Variable(torch.zeros(num_layers, bsz, dim_h)))
        _, hn = self.rnn(x, h0)
        hn2 = hn[-1] 
        y_cate_new = self.net_cate_new(hn2)
        y_color = self.net_color(hn2)
        y_gender = self.net_gender(hn2)
        y_sleeve = self.net_sleeve(hn2)
        return hn2, y_cate_new, y_color, y_gender, y_sleeve

############### training ###################

# model = get_variable(language_encoder())
# train_iter = []#for plot
# train_loss = []#for plot
# criterion = nn.CrossEntropyLoss()
# cuda_label_cate_new = get_variable(Variable(torch.LongTensor(bsz).zero_()))
# cuda_label_color = get_variable(Variable(torch.LongTensor(bsz).zero_()))
# cuda_label_gender = get_variable(Variable(torch.LongTensor(bsz).zero_()))
# cuda_label_sleeve = get_variable(Variable(torch.LongTensor(bsz).zero_()))

# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
# model.train()
# sampled_id = []
# for iter in range(30000):

#     #decrease learning rate
#     if iter == 10000:
#         optimizer = optim.SGD(model.parameters(), lr=0.001) 

#     if iter == 20000:
#         optimizer = optim.SGD(model.parameters(), lr=0.0003)
    
#     if iter == 25000:
#         optimizer = optim.SGD(model.parameters(), lr=0.0001)


#     assert bsz == 1

#     t = randint(0, m_train-1)
#     while t in sampled_id:
#         t = randint(0, m_train-1)
#     sampled_id.append(t)
#     sample_id = train_ind[t]
#     c = codeJ[sample_id][0]
#     l = len(c)
#     cuda_c_onehot = get_variable(torch.zeros(l, bsz, dim_voc))
#     for i in range(l):
#         cuda_c_onehot[i][0][int(c[i][0]-1)] = 1
#     cuda_c_onehot = Variable(cuda_c_onehot)

#     cuda_label_cate_new.data[0] = data_cate_new[sample_id][0]
#     cuda_label_color.data[0] = data_color[sample_id][0]
#     cuda_label_gender.data[0] = data_gender[sample_id][0]
#     cuda_label_sleeve.data[0] = data_sleeve[sample_id][0]

#     optimizer.zero_grad()
#     hn2, y_cate_new, y_color, y_gender, y_sleeve = model(cuda_c_onehot)
#     loss_cate_new = criterion(y_cate_new, cuda_label_cate_new)
#     loss_color = criterion(y_color, cuda_label_color)
#     loss_gender = criterion(y_gender, cuda_label_gender)
#     loss_sleeve = criterion(y_sleeve, cuda_label_sleeve)
#     loss = loss_cate_new + loss_color + loss_gender + loss_sleeve
#     if iter % 2000 == 0:
#         train_iter.append(iter)
#         train_loss.append(loss)
#     loss.backward()
#     optimizer.step()


#     if iter % 1000 == 0:
#         print('Training Iter %d: Loss = %.5f, cate_new (%.5f), color (%.5f), gender(%.5f), sleeve(%.5f)' % (iter, loss.data.item(), loss_cate_new.data.item(), loss_color.data.item(), loss_gender.data.item(), loss_sleeve.data.item()))

#     if iter % 10000 == 1:
#         torch.save(model.state_dict(), 'rnn_latest.pth')


#     # if iter % 1000 == 0:
#     #     plt.plot(train_iter, train_loss)
#     #     plt.show()
#     #     plt.pause(0.01)



# plt.plot(train_iter, train_loss)
# plt.show()

#################store####################
model = language_encoder()
model.load_state_dict(torch.load('src/language_encoder/rnn_latest.pth'))
encodes = np.zeros((78979,100))
print('Staring')

for i in range(len(codeJ)):
    if  i %5000 ==0:
        print((i/78979))
    c = codeJ[i][0]
    l = len(c)
    cuda_c_onehot = get_variable(torch.zeros(l, bsz, dim_voc))
    for j in range(l):
        cuda_c_onehot[j][0][int(c[j][0]-1)] = 1
        cuda_c_onehot = Variable(cuda_c_onehot,requires_grad=False)
    encode, a, b, c, d = model.forward(cuda_c_onehot)
    encodes[i]=encode.cpu().detach().numpy()
np.save('encode.npy',encodes)

print('Done')
# np.load('encode.npy')