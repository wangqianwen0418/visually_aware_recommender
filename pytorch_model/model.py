from io import StringIO, BytesIO
from PIL import Image
from scipy.misc import imresize
import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom
import os

import torch
import torchvision
from torch.autograd import Variable as V
from torchvision import transforms as trn


dataset_name = 'AmazonFashion6ImgPartitioned.npy'
dataset_dir = '/home/qianwen/KDD/dataset/amazon/'
dataset = np.load(dataset_dir + dataset_name, encoding = 'bytes')

[user_train, user_validation, user_test, Item, usernum, itemnum] = dataset
def preprocess(img_s, mean=np.array([0.43, 0.47, 0.49]), std=np.array([1.0, 1.0, 1.0])):
    return (imresize(np.asarray(Image.open(BytesIO(img_s))), (224, 224, 3)) / 255 - np.array(mean)) / np.array(std)

latent_d = 10
net = torchvision.models.resnet18(pretrained=True)
for param in net.parameters():
    param.require_grad = False
net.fc = torch.nn.Linear(512, latent_d)

class Siamese(torch.nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.net = net
        self.kernel = V(torch.randn(usernum, latent_d))
        
    def forward(self, input_i, input_j, input_index):
        output_i = self.net(input_i)
        output_j = self.net(input_j)
        u = torch.cat([self.kernel.select(0, k) for k in input_index]).resize(len(input_index), latent_d)
        output_i = output_i * u.expand_as(output_i)
        output_j = output_j * u.expand_as(output_j)
        return output_i, output_j

class DataGenerator():
    def __init__(self, source, batch_size, dim, mean=np.array([0.43, 0.47, 0.49], std=np.array([1.0, 1.0, 1.0]))):
        self.source = source
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.dim = dim

    def batches():
        while True:
            img_i = np.empty((self.batch_size, *self.dim))
            img_j = np.empty((self.batch_size, *self.dim))
            index = np.empty((self.batch_size), dtype=int)]
	    for k in range(self.batch_size):
	        u = random.randint(0, len(self.source)-1)
                index[k] = int(u)
                u_imgs = list(set([e[b'productid'] for e in self.source[u]]))
                 
                i = u_imgs[random.randint(0, len(u_imgs)-1)]
                img_i[k] = preprocess(Item[i][b'imgs'])
                j = random.randint(0, len(Item)-1)
                while j in u_imgs:
                    j = random.randint(0, len(Item)-1)
                img_j[k] = preprocess(Item[j][b'imgs'])
            yield img_i, img_j, index


siamese_net = Siamese()
traing_data = DataGenerator(source=user_train, batch_size=128, dim=(224, 224, 3))
iteration = 0
for batch in traing_data.batches():
    img_i, img_j, index = batch
    input_i = V(torch.from_numpy(img_i).float().permute(0,3,1,2))
    input_j = V(torch.from_numpy(img_j).float().permute(0,3,1,2))
    output_i, output_j = siamese_net(input_i, input_j, index)
    
    print()
'''
sample = (preprocess(Item[user_train[0][0][b'productid']][b'imgs']), preprocess(Item[10][b'imgs']), 0)
output_i, output_j = siamese_net( \
	V(torch.from_numpy(sample[0]).float().permute(2,0,1)).unsqueeze(0),\
	V(torch.from_numpy(sample[1]).float().permute(2,0,1)).unsqueeze(0),\
        [sample[2]])
print(output_i, output_j)
'''
