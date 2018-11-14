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
parser.add_argument("--kernel_wd", type=float, default=1e-2, 
                         help="weight decay of the kernel, default to 1e-2")
parser.add_argument("--linear_wd", type=float, default=1e-4,
                         help="weight decay of the linear layer, default to 1e-4")
args = parser.parse_args()
print(args.kernel_wd)
workdir = '/home/qianwen/KDD/cfree/ckpt/resnet_pretrained_50_{}_promoted'.format(args.kernel_wd)
if not os.path.exists(workdir):
    os.mkdir(workdir)

dataset_name = 'AmazonFashion6ImgPartitioned.npy'
dataset_dir = '/home/qianwen/KDD/dataset/'
dataset = np.load(dataset_dir + dataset_name, encoding = 'bytes')

[user_train, user_validation, user_test, Item, usernum, itemnum] = dataset

latent_d = 50
batch_size = 128


class Siamese(torch.nn.Module):
    def __init__(self, latent_d, batch_size, usernum):
        super(Siamese, self).__init__()
        net = torchvision.models.resnet18(pretrained=True)
        for param in net.parameters():
            param.requires_grad = False
        net.fc = torch.nn.Linear(512, latent_d)
        self.net = net
        self.kernel = torch.nn.Parameter(torch.randn(usernum, latent_d).cuda())
        
    def forward(self, input_i, input_j, input_index):
        output_i = self.net(input_i)
        output_j = self.net(input_j)
        u = torch.cat([self.kernel.select(0, k) for k in input_index]).resize(batch_size, latent_d)
        output_i = output_i * u
        output_j = output_j * u
        return output_i, output_j

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
       ])

class AmazonDataset(Dataset):    
    def __init__(self, source, transform):
        self.source = source
        self.transform = data_transform
        
    def __len__(self):
        return sum([len(v) for k,v in self.source.items()])
        
    def __getitem__(self, idx):   
        u = idx%usernum
        i = self.source[u][random.randint(0, len(self.source[u])-1)][b'productid']
        img_i = self.transform(Image.open(BytesIO(Item[i][b'imgs'])))
        u_imgs = list(set([e[b'productid'] for e in user_train[u]] +
                          [e[b'productid'] for e in user_validation[u]] +
                          [e[b'productid'] for e in user_test[u]]))
        if img_i.shape[0] == 1:
            img_i = img_i.repeat(3,1,1)
        j = i
        while j in u_imgs:
            j = random.randint(0, itemnum-1)
        img_j = self.transform(Image.open(BytesIO(Item[j][b'imgs'])))
        if img_j.shape[0] == 1:
            img_j = img_j.repeat(3,1,1)
        assert(img_i.shape[0] == 3 and img_j.shape[0] == 3)
        return u, img_i, img_j


def evaluate(model):
    scores = []
    validation_dataset = AmazonDataset(user_validation, transforms)
    validation_generator = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    model.eval()
    for index, input_i, input_j in validation_generator:
        input_i = input_i.cuda()
        input_j = input_j.cuda()
        output_i, output_j = model(input_i, input_j, index)
        scores += (torch.sum(output_i, 1) > torch.sum(output_j, 1)).data.tolist()
    model.train()
    return sum(scores) / len(scores)

siamese_net = Siamese(latent_d, batch_size, usernum)
siamese_net.cuda()
training_dataset = AmazonDataset(user_train, transforms)
training_generator = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
params = []
for name, param in siamese_net.named_parameters():
    if name == 'kernel' :
        params.append({'params': [param], 'weight_decay': args.kernel_wd, 'lr': 1e-3})
    elif name == "fc.weight" or name == "fc.bias":
        params.append({'params': [param], 'weight_decay': args.linear_wd, 'lr': 1e-3})
    else:
        params.append({'params': [param], 'weight_decay': 1e-4,'lr': 1e-4})
optimizer = torch.optim.Adam(params)

def learning_rate_schedule(optimizer, iteration):
    if iteration == 3:
        for param in optimizer.param_groups:
            param['lr'] = 1e-4
    elif iteration == 10:
        for param in optimizer.param_groups:
            param['lr'] = 1e-5

f = open(os.path.join(workdir, 'log.txt'), 'w')
max_iteration = 50
print("Training begin:")
for iteration in range(max_iteration):
    step = 0
    score = evaluate(siamese_net)
    print("Iteration {} Validation Score {}".format(iteration, score))
    f.write("Iteration {} Validation Score {}\n".format(iteration, score))
    f.flush()
    torch.save(siamese_net.state_dict(), os.path.join(workdir, 'iteration_{}'.format(iteration)))
    learning_rate_schedule(optimizer, iteration)
    for index, input_i, input_j in training_generator:
        input_i = input_i.cuda()
        input_j = input_j.cuda()
        output_i, output_j = siamese_net(input_i, input_j, index)
        distance = torch.sum(torch.nn.LogSigmoid()(torch.sum(output_i - output_j, 1)))
        score = round(float(torch.sum(torch.sum(output_i, 1) > torch.sum(output_j, 1))) / batch_size, 3)
        loss = -distance
        print('Iteration {}, Step {}, Loss {}, Score {}'.format(iteration, step, loss, score))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
f.close()

