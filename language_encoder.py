import torch.nn as nn
import torch
import numpy

dim_voc = 539 # size of vocabulary
num_layers = 2 
bsz = 1 #batch
dim_h = 100 # rnn hidden units
dim_cate_new = 19 # category of clothes
dim_color = 17 #color of clothes
dim_gender = 2 # gender
dim_sleeve = 4 # length of sleeve

#################CUDA###################
use_cuda = torch.cuda.is_available()
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

class language_encoder(nn.Module):
    def __init__(self):
        super(language_encoder, self).__init__()
        self.rnn = nn.RNN(dim_voc, dim_h, num_layers)
        self.net_cate_new = nn.Linear(dim_h, dim_cate_new)
        self.net_color = nn.Linear(dim_h, dim_color)
        self.net_gender = nn.Linear(dim_h, dim_gender)
        self.net_sleeve = nn.Linear(dim_h, dim_sleeve)

    def forward(self, x):
        h0 = get_variable(Variable(torch.zeros(num_layers, bsz, dim_h))
        output, hn = self.rnn(x, h0)
        hn2 = hn[-1] #output of the last layer in the last state
        y_cate_new = self.net_cate_new(hn2)
        y_color = self.net_color(hn2)
        y_gender = self.net_gender(hn2)
        y_sleeve = self.net_sleeve(hn2)
        return hn2, y_cate_new, y_color, y_gender, y_sleeve


model = language_encoder()
model.cuda()
criterion = nn.CrossEntropyLoss()
cuda_label_cate_new = get_variable(Variable(torch.LongTensor(bsz).zero_()))
cuda_label_color = get_variable(Variable(torch.LongTensor(bsz).zero_()))
cuda_label_gender = get_variable(Variable(torch.LongTensor(bsz).zero_()))
cuda_label_sleeve = get_variable(Variable(torch.LongTensor(bsz).zero_()))

optimizer = optim.SGD(model.parameters(), lr=0.01)
model.train()
for iter in range(1000000 * 10):
    #decrease learning rate
    if iter == 50000:
        optimizer = optim.SGD(model.parameters(), lr=0.001) 

    assert bsz == 1
    t = randint(0, m_train-1)
    sample_id = train_ind[t]
    c = codeJ[sample_id][0]
    l = len(c)
    cuda_c_onehot = torch.zeros(l, bsz, dim_voc).cuda()
    for i in range(l):
        cuda_c_onehot[i][0][int(c[i][0]-1)] = 1
    cuda_c_onehot = Variable(cuda_c_onehot)

    cuda_label_cate_new.data[0] = data_cate_new[sample_id][0]
    cuda_label_color.data[0] = data_color[sample_id][0]
    cuda_label_gender.data[0] = data_gender[sample_id][0]
    cuda_label_sleeve.data[0] = data_sleeve[sample_id][0]

    optimizer.zero_grad()
    hn2, y_cate_new, y_color, y_gender, y_sleeve = model(cuda_c_onehot)
    loss_cate_new = criterion(y_cate_new, cuda_label_cate_new)
    loss_color = criterion(y_color, cuda_label_color)
    loss_gender = criterion(y_gender, cuda_label_gender)
    loss_sleeve = criterion(y_sleeve, cuda_label_sleeve)
    loss = loss_cate_new + loss_color + loss_gender + loss_sleeve
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print('Training Iter %d: Loss = %.5f, cate_new (%.5f), color (%.5f), gender(%.5f), sleeve(%.5f)' % (iter, loss.data[0], loss_cate_new.data[0], loss_color.data[0], loss_gender.data[0], loss_sleeve.data[0]))

    if iter % 100000 == 1:
        torch.save(model.state_dict(), 'rnn_latest.pth')

    