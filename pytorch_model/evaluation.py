from io import StringIO, BytesIO
from PIL import Image
from scipy.misc import imresize
import random
import numpy as np
import os

import torch
import torchvision
from torch.autograd import Variable as V
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--workdir",  
                         help="work directory")
parser.add_argument("--iteration", type=int,
			help="load iteration")
args = parser.parse_args()
workdir = args.workdir

dataset_name = 'AmazonFashion6ImgPartitioned.npy'
dataset_dir = '/home/qianwen/KDD/dataset/'
dataset = np.load(dataset_dir + dataset_name, encoding = 'bytes')

[user_train, user_validation, user_test, Item, usernum, itemnum] = dataset

latent_d = 50
batch_size = 128


class One_branch_siamese(torch.nn.Module):
    def __init__(self, latent_d, batch_size, usernum):
        super(One_branch_siamese, self).__init__()
        net = torchvision.models.alexnet(pretrained=True)
        self.features = net.features
        self.classifier = torch.nn.Sequential(net.classifier[1], net.classifier[2], net.classifier[4], net.classifier[5]) 
        self.linear = torch.nn.Linear(4096, latent_d)
        self.kernel = torch.nn.Parameter(torch.randn(usernum, latent_d).cuda())
        
    def forward(self, input_i, input_index):
        output_i = self.classifier((self.features(input_i).resize(batch_size, 256*6*6)))
        output_i = self.linear(output_i)
        u = torch.cat([self.kernel.select(0, k) for k in input_index]).resize(batch_size, latent_d)
        output_i = output_i * u
        return output_i

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
       ])

class AmazonDataset(Dataset):    
    def __init__(self, itemset, transform):
        self.itemset = itemset
        self.transform = data_transform
        
    def __len__(self):
        return len(self.itemset)       
 
    def __getitem__(self, idx):
        img = self.transform(Image.open(BytesIO(self.itemset[idx][b'imgs'])))   
        if img.shape[0] == 1:
            img = img.repeat(3,1,1)
        return img, idx


def evaluate(model, u):
    scores = np.empty(len(Item))
    validation_dataset = AmazonDataset(Item, transforms)
    validation_generator = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
    model.eval()
    for input_i, ids in validation_generator:
        input_i = input_i.cuda()
        output_i = model(input_i, [u]*batch_size)
        output_scores = torch.sum(output_i, 1).data.tolist()
        for i in range(batch_size):
            scores[ids[i]] = output_scores[i]
    model.train()
    return scores

siamese_net = One_branch_siamese(latent_d, batch_size, usernum)
siamese_net.load_state_dict(torch.load(os.path.join(workdir, "iteration_{}".format(args.iteration))))
siamese_net.cuda()
workdir = os.path.join(workdir, "img")
os.mkdir(workdir)
print("Evaluation begin:")
for i in range(10):
    u = random.randint(0, usernum - 1)
    img_dir = os.path.join(workdir, "{}".format(u))
    os.mkdir(img_dir)
    I = evaluate(siamese_net, u)
    sortedargs = np.argsort(I)
    print(sortedargs[-10:])
    for img in user_train[u]:
        img_id = img[b'productid']
        Image.open(BytesIO(Item[img_id][b'imgs'])).save(os.path.join(img_dir, "{}_train.jpg".format(img_id)))
    for j in range(10):
        img_id = sortedargs[-j-1]
        Image.open(BytesIO(Item[img_id][b'imgs'])).save(os.path.join(img_dir, "{}_pos.jpg".format(img_id)))
    for j in range(10):
        img_id = sortedargs[j]
        Image.open(BytesIO(Item[img_id][b'imgs'])).save(os.path.join(img_dir, "{}_neg.jpg".format(img_id)))
    print("user {} has been recorded".format(u))

