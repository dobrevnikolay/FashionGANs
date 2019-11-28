import scipy.io
import os

lang_org = scipy.io.loadmat(os.path.join(os.path.dirname(__file__),'..','data','language_original.mat'))

# gender 0 corresponds to female
desc = lang_org['engJ']
text = str(desc[49][0][0])

print(text)