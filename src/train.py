import data_loader
import language_encoder
from torch.utils.data import DataLoader
import torch


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