import torch

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

LE = language_encoder()
LE.load_state_dict(torch.load('rnn_latest.pth'))#this should be modified
##############onehot and forward#####################

# x_id: the id of input description. Note: 
def encoding(X,x_id):
    c = X['codeJ'][x_id][0]
    l = len(c)
    cuda_c_onehot = get_variable(torch.zeros(l, bsz, dim_voc))
    for i in range(l):
        cuda_c_onehot[i][0][int(c[i][0]-1)] = 1
    cuda_c_onehot = Variable(cuda_c_onehot)
    hn2, _, _, _, _ = model(cuda_c_onehot)
return hn2

