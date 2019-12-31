from torch.utils.data import Dataset

def binary_representaiton(val, n_bits):
    binary_str = bin(val)[2:]
    while n_bits > len(binary_str):
            binary_str = '0'+binary_str
    arr =[]
    for i in range(len(binary_str)):
        arr.append(float(binary_str[i]))
    return arr


def create_mask_for_skin_tone(segmented_image):
    mask = copy.deepcopy(segmented_image)
    for i in range(7):
        if(2 == i or 6 == i):
            mask[i == mask] = 1
        else:
            mask[i==mask] = 0
    return mask

def apply_mask(segmented_image,real_image):
    mask = create_mask_for_skin_tone(segmented_image)
    masked = copy.deepcopy(real_image)
    for channel in range(len(real_image)):
        masked[channel] = np.multiply(masked[channel],mask)
    return masked


class FashionData(Dataset):
    def __init__(self,X,y,type_of_data):
        self.X = X[type_of_data]
        self.y = y[type_of_data]
    
    def __getitem__(self, index):
        design_encoding = []
        design_encoding.append(float(self.X['gender'][index]))
        design_encoding.extend(binary_representaiton(self.X['color'][index],5))
        design_encoding.extend(binary_representaiton(self.X['sleeve'][index],3))
        design_encoding.extend(binary_representaiton(self.X['cate_new'][index],5))
        design_encoding.append(self.X['r'][index])
        design_encoding.append(self.X['g'][index])
        design_encoding.append(self.X['b'][index])
        design_encoding.append(self.X['y'][index])
        design_encoding.extend(self.X['encoding'][index])
        design_encoding = np.array(design_encoding)

        #return (self.X['segmented_image'][index],self.y[index])
        return (design_encoding,self.X['down_sampled_images'][index],get_segmented_image_7(self.X['segmented_image'][index]),self.y[index])        

    def __len__(self):
        return len(self.y)



# should we normalize the real_images
def construct_data(segmented_images,real_images,indeces,language,encoded_values):
    X = {}
    y = {}

    X['train'] = {}
    X['train']['gender'] =[]
    X['train']['color'] =[]
    X['train']['sleeve'] =[]
    X['train']['cate_new'] =[]
    X['train']['segmented_image'] = []
    X['train']['down_sampled_images'] = []
    X['train']['description'] = []
    X['train']['encoding'] = []
    X['train']['codeJ'] = []
    X['train']['r'] = []
    X['train']['g'] = []
    X['train']['b'] = []
    X['train']['y'] = []

    X['test'] = {}
    X['test']['gender'] =[]
    X['test']['color'] =[]
    X['test']['sleeve'] =[]
    X['test']['cate_new'] =[]
    X['test']['segmented_image'] = []
    X['test']['down_sampled_images'] = []
    X['test']['description'] = []
    X['test']['encoding'] = []
    X['test']['codeJ'] = []
    X['test']['r'] = []
    X['test']['g'] = []
    X['test']['b'] = []
    X['test']['y'] = []

    y['train'] = []
    y['test'] = []

    # length_to_iterate_train = len(indeces['train_ind'])
    length_to_iterate_train = 10000
    # length_to_iterate_test = len(indeces['test_ind'])
    length_to_iterate_test = 1000


    for i in range(length_to_iterate_train):
        idx = indeces['train_ind'][i][0] - 1
        X['train']['gender'].append(language['gender_'][idx][0])
        X['train']['color'].append(language['color_'][idx][0])
        X['train']['sleeve'].append(language['sleeve_'][idx][0])
        X['train']['cate_new'].append(language['cate_new'][idx][0])
        X['train']['description'].append(str(language['engJ'][idx][0][0]))
        X['train']['encoding'].append(encoded_values[idx])
        # X['train']['segmented_image'].append(segmented_images[idx])
        X['train']['segmented_image'].append(get_segmented_image_7(segmented_images[idx]))  
        X['train']['codeJ'].append(str(language['codeJ'][idx][0][0]))
        skin_tone = apply_mask(np.reshape(segmented_images[idx],(128,128)),real_images[idx])

        r,g,b = np.median(skin_tone[0]), np.median(skin_tone[1]), np.median(skin_tone[2])

        X['train']['r'].append(r)
        X['train']['g'].append(g)
        X['train']['b'].append(b)
        X['train']['y'].append(0.2125*r + 0.7154*g +  0.0721*b)
        #X['train']['down_sampled_images'].append(get_downsampled_image(segmented_images[idx][0]))
        X['train']['down_sampled_images'].append(get_downsampled_image_4(segmented_images[idx]))
        y['train'].append(real_images[idx])

    for i in range(length_to_iterate_test):
        idx = indeces['test_ind'][i][0] - 1
        X['test']['gender'].append(language['gender_'][idx][0])
        X['test']['color'].append(language['color_'][idx][0])
        X['test']['sleeve'].append(language['sleeve_'][idx][0])
        X['test']['cate_new'].append(language['cate_new'][idx][0])
        X['test']['description'].append(str(language['engJ'][idx][0][0]))
        X['test']['encoding'].append(encoded_values[idx])
        X['test']['segmented_image'].append(get_segmented_image_7(segmented_images[idx]))  
        # X['test']['segmented_image'].append(segmented_images[idx])
        X['test']['codeJ'].append(str(language['codeJ'][idx][0][0]))
        skin_tone = apply_mask(np.reshape(segmented_images[idx],(128,128)),real_images[idx])

        r,g,b = np.median(skin_tone[0]), np.median(skin_tone[1]), np.median(skin_tone[2])

        X['test']['r'].append(r)
        X['test']['g'].append(g)
        X['test']['b'].append(b)
        X['test']['y'].append(0.2125*r + 0.7154*g +  0.0721*b)

        X['test']['down_sampled_images'].append(get_downsampled_image_4(segmented_images[idx]))

        y['test'].append(real_images[idx])
    
    return (X,y)

    
def normalize_pictures(real_images):
    for image in real_images:
        for channel in range(len(image)):
            image[channel] = (image[channel] - image[channel].min()) / (image[channel].max()-image[channel].min())
    return real_images


def load_data():

    segmented_images = None
    real_images = None
    print("Check if the serialized data is present")
    # check if the serialized images are present if not create them
    if not(os.path.isfile(segmented_images_raw_path) and os.path.isfile(real_images_raw_path)):
        print("We have to read the h5 file, it will take time")
        with h5py.File(h5_file_path, 'r') as f:   

            # Get the data
            # Segmentated images 1x 128x 128 values from 0 to 6
            segmented_images = list(f['b_'])
            # Real images three channels instad of 0-255 values for a pixel we have normalized values between [-1;1]
            pickle.dump(segmented_images, open(segmented_images_raw_path, 'wb')) 
            real_images = list(f['ih'])
            #normalize the real images
            real_images = normalize_pictures(real_images)
            pickle.dump(real_images, open(real_images_raw_path, 'wb')) 
            print("H5 read and data has been serialized")
    if None == segmented_images:
        infile = open(segmented_images_raw_path,'rb')
        segmented_images = pickle.load(infile)
        infile.close()

    if None == real_images:
        real_images = pickle.load(open(real_images_raw_path,'rb'))
    print("Images has been loaded successfully")
    print("Now reading .mat files")
    # now read language
    lang_org = scipy.io.loadmat(language_original_path)

    # read the indeces as well
    indeces = scipy.io.loadmat(indeces_path)

    print("Everything is loaded now constructing the dictionaries")
    
    encoded_values = np.load(lang_encoding)

    (X,y) = construct_data(segmented_images,real_images,indeces,lang_org, encoded_values)
    print("Data constructed")
    print("Pickle the data")
    handle = open(os.path.join(os.path.dirname(__file__),'..','data','data.pkl'),'wb')
    pickle.dump((X,y), handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    
    return (X,y)