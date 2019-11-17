import scipy.io

lang_org = scipy.io.loadmat("../data/language_original.mat")

# gender 0 corresponds to female
desc = lang_org['engJ']
text = str(desc[49][0][0])

print(text)